import os
import sys
import itertools
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# 在文件顶部确保有这个导入
from peft import PeftModel
import deepspeed # 其他必要的导入
from torch import distributed as dist
from transformers import PreTrainedModel
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME


from peft import PeftModel
from transformers import CONFIG_NAME, WEIGHTS_NAME
from transformers import Trainer, get_scheduler
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from transformers.deepspeed import HfDeepSpeedConfig, deepspeed_load_checkpoint

from tqdm import tqdm
from typing import (
    List, Any, Dict,
    Optional, Union,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from llava_dpo.logger import Logger

from llava_dpo.constants import ADAM_BETAS, IMAGE_TOKEN_INDEX, IGNORE_INDEX, ASSISTANT_TOKEN_IDS
from llava_dpo.model.utils import gather_log_probabilities, gather_log_probabilities_for_cfg
from llava_dpo.model import LlavaLlamaForCausalLM
from llava_dpo.utils import (
    is_main_process, 
    to_device, 
    get_all_reduce_mean, 
    get_all_reduce_max,
    get_indexes, 
    calculate_log_probs, 
    get_log_probs, 
    sample_random_image,
    is_multi_turn,
    get_answer_index,
)


def maybe_zero_3(param, ignore_status=False, name=None):
    """
    该函数用于处理可能处于 ZeRO-3 格式的参数。
    如果参数处于 ZeRO-3 格式，则会收集参数数据；
    如果不是 ZeRO-3 格式，则直接 detach 并 clone 数据。
    参数:
        param: 需要处理的参数。
        ignore_status: 若为 True，则忽略参数的状态。
        name: 参数名称，用于日志记录。
    返回:
        处理后的参数数据。
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class DummyDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    一个具有固定长度并返回空字典的虚拟数据集（DummyDataset）。
    当实际数据集不可用时，可用此类创建 DataLoader。
    """
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {}


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        设置优化器。

        我们提供了一个合理的默认设置，通常效果很好。如果你想使用其他优化器，可以在 Trainer 初始化时通过 `optimizers` 传入一个元组，或者通过子类化并重写此方法实现自定义。
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


class DPOLLaVATrainer(LLaVATrainer, Trainer):
    
    model: deepspeed.DeepSpeedEngine
    reference_model: deepspeed.DeepSpeedEngine
    
    def __init__(
        self, args,
        model: Union[PreTrainedModel, nn.Module],
        ref_model: Union[PreTrainedModel, nn.Module],
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
        data_collator: Optional[DataCollator] = None,
        ptx_data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ptx_train_dataset: Optional[Dataset] = None,
        ptx_eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        args.num_train_epochs = int(args.num_train_epochs)
        self.args = args
        self.scale_coeff = self.args.scale_coeff
        self.label_smoothing = self.args.label_smoothing
        self.dynamic_label_smoothing = self.args.dynamic_label_smoothing
        self.language_bias_reduce = self.args.language_bias_reduce
        
        # self.args.need_eval = eval_dataset is not None
        self.logger = Logger(log_project=self.args.log_project, log_dir=self.args.output_dir)
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=args.per_device_train_batch_size,
        )
        
        self.use_ptx = ptx_train_dataset is not None
        if self.use_ptx:
            self.ptx_train_dataloader = DataLoader(
                ptx_train_dataset,
                collate_fn=ptx_data_collator,
                sampler=DistributedSampler(train_dataset, shuffle=True), # 分布式采样DistributedSampler(ptx_train_dataset, shuffle=True),
                batch_size=args.per_device_ptx_train_batch_size,
            )
        else:
            self.ptx_train_dataloader = DataLoader(DummyDataset(len(self.train_dataloader)))
        
    
        self.args.num_update_steps_per_epoch = ( 
            len(self.train_dataloader) + self.args.gradient_accumulation_steps - 1
        ) // self.args.gradient_accumulation_steps  # 向上取整操作
        self.args.total_training_steps = self.args.num_train_epochs * self.args.num_update_steps_per_epoch
        # 如果使用了 PTX 数据集，则将训练批次大小翻倍
        if self.use_ptx:
            self.args.gradient_accumulation_steps *= 2
            ds_train_config['train_batch_size'] *= 2
            ds_train_config['gradient_accumulation_steps'] *= 2
        
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)
        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)
        
        self.model = model
        self.create_optimizer_and_scheduler(num_training_steps=self.args.total_training_steps)            
        
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.args,
            config=self.ds_train_config,
            lr_scheduler=self.lr_scheduler,
            dist_init_required=True,
        )
        if self.args.resume_from_ckpt:
            deepspeed_load_checkpoint(self.model, self.args.resume_from_ckpt)
        if self.args.gradient_checkpointing: #好像有点问题
            self.model.gradient_checkpointing_enable()
        
        self.tokenizer = tokenizer
        
        self.reference_model, *_ = deepspeed.initialize(
            model=ref_model,
            config=ds_eval_config,
        )
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None:
            projector_parameters = [name for name, _ in self.model.named_parameters() if "mm_projector" in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_projector_lr,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        
        if (
            self.ds_train_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            self.optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=ADAM_BETAS,
                # **optimizer_kwargs
            )
        else:
            self.optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=ADAM_BETAS,
            )
        
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        设置学习率调度器。Trainer 的优化器必须在调用此方法前已设置好，或者通过参数传入。

        参数:
            num_training_steps (int): 训练的总步数。
        """
        num_warmup_steps = int(self.args.warmup_ratio * self.args.total_training_steps)
        # self.ds_train_config['scheduler']['params']['warmup_num_steps'] = num_warmup_steps
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self._created_lr_scheduler = True
        return self.lr_scheduler
    
    def compute_log_probs(
        self,
        model: LlavaLlamaForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor,
        images: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences.该函数输入文本、标签和（可选）图像，
        输出每个 token 的log probability（对数概率），可选返回 logits，常用于对比损失或奖励计算。"""
        logits = model(input_ids, attention_mask=attention_mask, images=images).logits

        return gather_log_probabilities(logits[:, :-1], labels[:, 1:]), \
            (logits[:, :-1] if self.args.gamma != 1 and self.args.use_logits else None)

# def gather_log_probabilities(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
#     """Gather log probabilities of the given labels from the logits."""
#     log_probs = F.log_softmax(logits.float(), dim=-1)
#     log_probs_labels = log_probs.gather(dim=-1, index=labels.long().unsqueeze(dim=-1))
#     return log_probs_labels.squeeze(dim=-1)
# 这段代码的作用是：
# 给定模型输出的 logits（未归一化的分数）和对应的标签 labels，计算每个标签对应的对数概率（log probability）。

# 具体步骤如下：

# F.log_softmax(logits.float(), dim=-1)：先对 logits 做 softmax，然后取对数，得到每个类别的对数概率。
# log_probs.gather(dim=-1, index=labels.long().unsqueeze(dim=-1))：根据 labels，从 log_probs 中选出每个样本对应标签的对数概率。
# squeeze(dim=-1)：去掉多余的维度，返回一维的对数概率张量。
# 简单来说，这个函数就是用来获取每个样本标签在模型输出中的对数概率，常用于计算损失函数（如交叉熵损失）。

    @torch.no_grad()
    def eval_step(
        self,
        better_input_ids: torch.LongTensor,
        better_labels: torch.LongTensor,
        better_attention_mask: torch.BoolTensor,
        worse_input_ids: torch.LongTensor,
        worse_labels: torch.LongTensor,
        worse_attention_mask: torch.BoolTensor,
        images: torch.Tensor,
        category_ids: torch.Tensor = None,
        confidence: torch.Tensor = None,
    ) -> tuple[tuple]:
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        
        input_ids = torch.cat([better_input_ids, worse_input_ids], dim=0)
        attention_mask = torch.cat([better_attention_mask, worse_attention_mask], dim=0)
        labels = torch.cat([better_labels, worse_labels], dim=0)
        if images is not None:
            better_images, worse_images = images[:,0,:,:,:], images[:,1,:,:,:]
            images = torch.cat([better_images, worse_images], dim=0)
        
        label_mask = torch.logical_and(labels.ne(IMAGE_TOKEN_INDEX), labels.ne(IGNORE_INDEX))
        labels = (labels * label_mask).long()
        label_mask = label_mask[:, 1:]
        
        sequence_log_probs, _ = self.compute_log_probs(
            self.model.module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
        )  # size = (2 * B, L - 1)
        randimg_sequence_log_probs_list = []
        for _ in range(self.args.n_random_images):
            fake_images = torch.stack([
                sample_random_image(images.shape[1:]) for _ in range(images.size(0))
            ], dim=0).to(images.device)
            randimg_sequence_log_probs, _ = self.compute_log_probs(
                self.model.module,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=fake_images,
            )  # size = (2 * B, L - 1)
            randimg_sequence_log_probs_list.append(randimg_sequence_log_probs)
        if self.args.n_random_images == 0:
            fake_images = torch.zeros_like(images).to(images.device)
            randimg_sequence_log_probs, _ = self.compute_log_probs(
                self.model.module,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=fake_images,
            )  # size = (2 * B, L - 1)
            randimg_sequence_log_probs_list.append(randimg_sequence_log_probs)
        randimg_sequence_log_probs = torch.stack(randimg_sequence_log_probs_list, dim=-1).mean(dim=-1)
        
        return (sequence_log_probs, randimg_sequence_log_probs), (input_ids, label_mask)
    
    def train_step(
        self,
        better_input_ids: torch.LongTensor,
        better_labels: torch.LongTensor,
        better_attention_mask: torch.BoolTensor,
        worse_input_ids: torch.LongTensor,
        worse_labels: torch.LongTensor,
        worse_attention_mask: torch.BoolTensor,
        images: torch.Tensor,
        category_ids: torch.Tensor = None,
        confidence: torch.Tensor = None,
    ) -> dict[str, Any]:
        
        # 确保模型在训练模式
        self.model.train()

        # ========================================================================
        # 步骤一：提取交叉注意力图 (此步骤不更新模型权重)
        # ========================================================================

        with torch.enable_grad():
            ATT_LAYER = 14
            NUM_IMG_TOKENS = 576
            NUM_PATCHES = 24

            batch_size = better_input_ids.size(0)
            input_ids_list = better_input_ids.tolist()
            # 找到每个样本中 <image> token 的起始位置
            image_start_pos_list = [ids.index(IMAGE_TOKEN_INDEX) for ids in input_ids_list]
            
            # 1. 设置开关以获取注意力
            self.model.module.config.output_attentions = True
            # 注意：这个 standard_attention_layer_idx 可能是您自定义的配置，请确保其有效
            self.model.module.config.standard_attention_layer_idx = ATT_LAYER
            
            # 2. 第一次前向传播，仅为获取注意力梯度
            outputs_for_attn = self.model.module(
                better_input_ids, 
                attention_mask=better_attention_mask, 
                images=images[:,0,:,:,:], 
                output_attentions=True
            )
            
            # 找到每个样本中第一个需要预测的token的索引 (您的实现是正确的)
            first_output_token_indices = (better_labels != IGNORE_INDEX).float().argmax(dim=1)
            
            # 获取模型在这些位置上的预测logits
            batch_indices = torch.arange(batch_size, device=outputs_for_attn.logits.device)
            first_output_logits = outputs_for_attn.logits[batch_indices, first_output_token_indices, :]
            
            # 创建代理loss，用于反向传播以获取梯度
            # 使用模型自己的最高预测作为目标，这是一种常见做法
            pseudo_labels = torch.argmax(first_output_logits, dim=1)
            proxy_loss = -nn.functional.cross_entropy(first_output_logits, pseudo_labels)
            
            # 3. 获取目标层的注意力张量
            attentions = outputs_for_attn.attentions[ATT_LAYER]
            assert attentions is not None and attentions.requires_grad

            # 4. 计算注意力梯度
            attention_grads = torch.autograd.grad(proxy_loss, attentions, retain_graph=False)[0]
            
            # 使用梯度加权，并应用ReLU来只保留正向贡献
            grad_att = attentions * F.relu(attention_grads)
            
            # 5. [修正部分] 计算并保存每个样本的注意力图
            att_maps = []
            for i in range(batch_size):
                # 当前样本的图像token起始位置
                image_start_pos = image_start_pos_list[i]
                
                # 当前样本中我们关心的“第一个输出token”的索引
                query_token_idx = first_output_token_indices[i].item()

                # 正确切片: Query来自 query_token_idx, Key是图像区域
                att_map_slice = grad_att[i, :, query_token_idx, image_start_pos : image_start_pos + NUM_IMG_TOKENS]
                
                # 在所有注意力头上取平均
                att_map = att_map_slice.mean(dim=0)
                
                # 重塑并保存
                att_map = att_map.to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
                att_maps.append(att_map)
            
        # ========================================================================
        # 步骤二：清理中间状态，为正式训练做准备
        # ========================================================================

        del outputs_for_attn, attentions, attention_grads, grad_att, proxy_loss, first_output_logits
        self.model.module.config.output_attentions = False 

        bbox_size = 336

# ... 后续代码 ...
        """
        DPO 算法的损失函数。

        参数:
            better_input_ids (torch.LongTensor): 更优答案的输入 ID。
            better_attention_mask (torch.BoolTensor): 更优答案的 attention mask。
            worse_input_ids (torch.LongTensor): 较差答案的输入 ID。
            worse_attention_mask (torch.BoolTensor): 较差答案的 attention mask。
            images (torch.Tensor): 图像张量。

        返回:
            dict[str, torch.Tensor]: 包含 loss、更优样本奖励、较差样本奖励等信息的字典。
        """
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        batch_size = better_input_ids.size(0)
        
        input_ids = torch.cat([better_input_ids, worse_input_ids], dim=0)
        attention_mask = torch.cat([better_attention_mask, worse_attention_mask], dim=0)
        labels = torch.cat([better_labels, worse_labels], dim=0)
        if images is not None:
            better_images, worse_images = images[:,0,:,:,:], images[:,1,:,:,:]
            images = torch.cat([better_images, worse_images], dim=0)
        
        label_mask = torch.logical_and(labels.ne(IMAGE_TOKEN_INDEX), labels.ne(IGNORE_INDEX))
        labels = (labels * label_mask).long()
        better_label_mask, worse_label_mask = label_mask[:, 1:].chunk(chunks=2, dim=0)
        
        sequence_log_probs, sequence_logits  = self.compute_log_probs(
            self.model.module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
        )  # size = (2 * B, L - 1)
        better_log_probs, worse_log_probs = sequence_log_probs.chunk(chunks=2, dim=0)   # size = (B, L - 1)
        
        self.reference_model.eval()
        with torch.no_grad():
            ref_sequence_log_probs, _ = self.compute_log_probs(
                self.reference_model.module,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images,
            )  # size = (2 * B, L - 1)
            ref_better_log_probs, ref_worse_log_probs = ref_sequence_log_probs.chunk(chunks=2, dim=0)  # size = (B, L - 1)
        
        randimg_better_log_probs_list, randimg_worse_log_probs_list = [], []
        ref_randimg_better_log_probs_list, ref_randimg_worse_log_probs_list = [], []
        randimg_logits_list = []
        for _ in range(self.args.n_random_images + 1):
            if len(randimg_better_log_probs_list) >= self.args.n_random_images > 0: break
            fake_images = torch.zeros_like(images).to(images.device) if self.args.n_random_images == 0 else \
                torch.stack([
                    sample_random_image(images.shape[1:]) for _ in range(images.size(0))
                ], dim=0).to(images.device)
            torch.cuda.empty_cache()
            with torch.no_grad():
                randimg_sequence_log_probs, randimg_sequence_logits = self.compute_log_probs(
                    self.model.module,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=fake_images,
                )  # size = (2 * B, L - 1)
            with torch.no_grad():
                ref_randimg_sequence_log_probs, ref_randimg_sequence_logits = self.compute_log_probs(
                    self.reference_model.module,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=fake_images,
                )
            if not self.args.dynamic_llm:
                randimg_sequence_logits = ref_randimg_sequence_logits
            randimg_better_log_probs, randimg_worse_log_probs = randimg_sequence_log_probs.chunk(chunks=2, dim=0)  # size = (B, L - 1)
            randimg_better_log_probs_list.append(randimg_better_log_probs)
            randimg_worse_log_probs_list.append(randimg_worse_log_probs)
            ref_randimg_better_log_probs, ref_randimg_worse_log_probs = ref_randimg_sequence_log_probs.chunk(chunks=2, dim=0)  # size = (B, L - 1)
            ref_randimg_better_log_probs_list.append(ref_randimg_better_log_probs)
            ref_randimg_worse_log_probs_list.append(ref_randimg_worse_log_probs)
            if self.args.use_logits and self.args.gamma != 1:
                randimg_logits_list.append(randimg_sequence_logits)
        randimg_better_log_probs = torch.stack(randimg_better_log_probs_list, dim=-1).mean(dim=-1)
        randimg_worse_log_probs = torch.stack(randimg_worse_log_probs_list, dim=-1).mean(dim=-1)
        ref_randimg_better_log_probs = torch.stack(ref_randimg_better_log_probs_list, dim=-1).mean(dim=-1)
        ref_randimg_worse_log_probs = torch.stack(ref_randimg_worse_log_probs_list, dim=-1).mean(dim=-1)
        if self.args.use_logits and self.args.gamma != 1:
            randimg_sequence_logits = torch.stack(randimg_logits_list, dim=-1).mean(dim=-1)
        
        if self.args.use_logits:
            cfg_log_probs = gather_log_probabilities_for_cfg(sequence_logits, randimg_sequence_logits, labels[:, 1:], gamma=self.args.gamma)
            cfg_better_log_probs, cfg_worse_log_probs = cfg_log_probs.chunk(chunks=2, dim=0)
        
        losses = []
        better_sample_rewards, worse_sample_rewards = [], []
        better_img_contributions_rand, worse_img_contributions_rand = [], []
        coef_list = []

        print(f"[DPO] batch size: {batch_size}, n_random_images: {self.args.n_random_images}")
        for i in range(batch_size):
            equal_text = torch.all(torch.eq(input_ids[i], worse_input_ids[i])).item()
            equal_img = torch.all(torch.eq(better_images[i], worse_images[i])).item()
            equal_lable = torch.all(torch.eq(better_labels[i], worse_labels[i])).item()

            if equal_lable:
                print(f"[Warning] Sample {i} has identical labels in better/worse pairs.")
            else:
                print(f"[Info] Sample {i} has different labels in better/worse pairs.")
            if equal_text:
                print(f"[Warning] Sample {i} has identical text in better/worse pairs.")
            if equal_img:
                print(f"[Warning] Sample {i} has identical images in better/worse pairs.")
            
            coeff = 1
            if category_ids[i].eq(0):
                coeff = self.args.instruct_coef
            if category_ids[i].eq(1):
                coeff = self.args.region_coef
            if category_ids[i].eq(2):
                coeff = self.args.vqa_coef
            
            # language bias pairs
            better_start_index, better_end_index = get_answer_index(better_input_ids[i], final_answer=True), better_attention_mask[i].nonzero()[-1]
            better_seq_slice = slice(better_start_index - 1, better_end_index)
            ith_better_log_probs = calculate_log_probs(better_log_probs[i, better_seq_slice], better_label_mask[i, better_seq_slice], return_average=self.args.ipo)
            ith_ref_better_log_probs = calculate_log_probs(ref_better_log_probs[i, better_seq_slice], better_label_mask[i, better_seq_slice], return_average=self.args.ipo)
            ith_rimg_better_log_probs = calculate_log_probs(randimg_better_log_probs[i, better_seq_slice].detach(), better_label_mask[i, better_seq_slice], return_average=self.args.ipo)
            ith_ref_rimg_better_log_probs = calculate_log_probs(ref_randimg_better_log_probs[i, better_seq_slice], better_label_mask[i, better_seq_slice], return_average=self.args.ipo)
            better_img_contributions_rand.append(ith_better_log_probs - ith_rimg_better_log_probs)
            better_log_ratio = ith_better_log_probs - ith_ref_better_log_probs
            better_rand_logits = better_log_ratio - (ith_rimg_better_log_probs - ith_ref_rimg_better_log_probs)
            
            worse_start_index, worse_end_index = get_answer_index(worse_input_ids[i], final_answer=True), worse_attention_mask[i].nonzero()[-1]
            worse_seq_slice = slice(worse_start_index - 1, worse_end_index)
            ith_worse_log_probs = calculate_log_probs(worse_log_probs[i, worse_seq_slice], worse_label_mask[i, worse_seq_slice], return_average=self.args.ipo)
            ith_ref_worse_log_probs = calculate_log_probs(ref_worse_log_probs[i, worse_seq_slice], worse_label_mask[i, worse_seq_slice], return_average=self.args.ipo)
            ith_rimg_worse_log_probs = calculate_log_probs(randimg_worse_log_probs[i, worse_seq_slice].detach(), worse_label_mask[i, worse_seq_slice], return_average=self.args.ipo)
            ith_ref_rimg_worse_log_probs = calculate_log_probs(ref_randimg_worse_log_probs[i, worse_seq_slice], worse_label_mask[i, worse_seq_slice], return_average=self.args.ipo)
            worse_img_contributions_rand.append(ith_worse_log_probs - ith_rimg_worse_log_probs)
            worse_log_ratio = ith_worse_log_probs - ith_ref_worse_log_probs
            
            # better <-> worse
            if not self.args.ipo and self.args.gamma == 1:
                diverge_index, better_end_index, worse_end_index = get_indexes(
                    better_input_ids[i], worse_input_ids[i],
                    better_attention_mask[i], worse_attention_mask[i],
                )
                better_seq_slice = slice(diverge_index - 1, better_end_index) #切片描述符
                worse_seq_slice = slice(diverge_index - 1, worse_end_index)
            
            ith_ref_better_log_probs = calculate_log_probs(ref_better_log_probs[i, better_seq_slice], better_label_mask[i, better_seq_slice], return_average=self.args.ipo)
            ith_ref_worse_log_probs = calculate_log_probs(ref_worse_log_probs[i, worse_seq_slice], worse_label_mask[i, worse_seq_slice], return_average=self.args.ipo)
            if self.args.use_logits:
                ith_better_log_probs = calculate_log_probs(cfg_better_log_probs[i, better_seq_slice], better_label_mask[i, better_seq_slice], return_average=self.args.ipo)
                ith_worse_log_probs = calculate_log_probs(cfg_worse_log_probs[i, worse_seq_slice], worse_label_mask[i, worse_seq_slice], return_average=self.args.ipo)
            else:
                ith_better_log_probs = calculate_log_probs(better_log_probs[i, better_seq_slice], better_label_mask[i, better_seq_slice], 
                                                           randimg_log_probs=randimg_better_log_probs[i, better_seq_slice].detach() if self.args.dynamic_llm else ref_randimg_better_log_probs[i, better_seq_slice],
                                                           return_average=self.args.ipo, gamma=self.args.gamma)
                ith_worse_log_probs = calculate_log_probs(worse_log_probs[i, worse_seq_slice], worse_label_mask[i, worse_seq_slice], 
                                                          randimg_log_probs=randimg_worse_log_probs[i, worse_seq_slice].detach() if self.args.dynamic_llm else ref_randimg_worse_log_probs[i, worse_seq_slice],
                                                          return_average=self.args.ipo, gamma=self.args.gamma)
            logits = (ith_better_log_probs - ith_ref_better_log_probs) - (ith_worse_log_probs - ith_ref_worse_log_probs)
            
            coef1 = (better_log_probs - (ref_randimg_better_log_probs if self.args.importance_sampling == 'vl' else ref_better_log_probs))[i, better_seq_slice].sum()
            coef2 = (worse_log_probs - (ref_randimg_worse_log_probs if self.args.importance_sampling == 'vl' else ref_worse_log_probs))[i, worse_seq_slice].sum()
            coef_list.append((coef1 + coef2).detach())
            if self.args.importance_sampling:
                coeff *= (coef1 + coef2).clamp(min=-1, max=1).exp().detach()
            
            if self.args.ipo:
                losses.append(coeff * ((logits - 1 / (2 * self.scale_coeff)) ** 2))
                if self.language_bias_reduce and not images[i].eq(0).all():
                    losses.append(coeff * ((better_rand_logits - 1 / (2 * self.scale_coeff)) ** 2))
            else:
                if self.dynamic_label_smoothing:
                    label_smoothing = .2 if self.args.not_dynamic else (1 - confidence[i]) / 2
                else:
                    label_smoothing = self.label_smoothing
                losses.append(coeff * (
                    - F.logsigmoid(self.scale_coeff * logits) * (1 - label_smoothing)
                    - F.logsigmoid(-self.scale_coeff * logits) * label_smoothing
                ))
                if self.language_bias_reduce and not images[i].eq(0).all():
                    _label_smoothing = label_smoothing / 2
                    losses.append(coeff * (
                        - F.logsigmoid(self.scale_coeff * better_rand_logits) * (1 - _label_smoothing)
                        - F.logsigmoid(-self.scale_coeff * better_rand_logits) * _label_smoothing
                    ))
            
            better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
            worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())
        
        loss = torch.stack(losses).mean()
        
        better_sample_rewards = torch.stack(better_sample_rewards)  # size = (B,)
        worse_sample_rewards = torch.stack(worse_sample_rewards)  # size = (B,)
        rewards_accuracy = (
            (better_sample_rewards > worse_sample_rewards).float().mean()
        )
        
        better_sample_rewards = better_sample_rewards.mean()  # size = ()
        worse_sample_rewards = worse_sample_rewards.mean()  # size = ()
        rewards_margin = better_sample_rewards - worse_sample_rewards
        better_img_contributions_rand = torch.stack(better_img_contributions_rand).mean()
        worse_img_contributions_rand = torch.stack(worse_img_contributions_rand).mean()
        max_coef = torch.stack(coef_list, dim=0).max()
        avg_coef = torch.stack(coef_list, dim=0).mean()
        
        self.model.backward(loss)
        self.model.step()
        
        loss = get_all_reduce_mean(loss)
        better_sample_rewards = get_all_reduce_mean(better_sample_rewards)
        worse_sample_rewards = get_all_reduce_mean(worse_sample_rewards)
        rewards_accuracy = get_all_reduce_mean(rewards_accuracy)
        rewards_margin = get_all_reduce_mean(rewards_margin)
        better_img_contributions_rand = get_all_reduce_mean(better_img_contributions_rand)
        worse_img_contributions_rand = get_all_reduce_mean(worse_img_contributions_rand)
        max_coef = get_all_reduce_max(max_coef)
        avg_coef = get_all_reduce_mean(avg_coef)
        
        return {
            'train/loss': loss.item(),
            'train/better_sample_rewards': better_sample_rewards.item(),
            'train/worse_sample_rewards': worse_sample_rewards.item(),
            'train/rewards_accuracy': rewards_accuracy.item(),
            'train/rewards_margin': rewards_margin.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
            'train/better_img_contributions_rand': better_img_contributions_rand.item(),
            'train/worse_img_contributions_rand': worse_img_contributions_rand.item(),
            'train/max_coef': max_coef.item(),
            'train/avg_coef': avg_coef.item(),
        }
    
    def ptx_step(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        images: torch.Tensor,
    ) -> dict[str, Any]:
        outputs = self.model.module(input_ids, attention_mask=attention_mask, labels=labels, images=images)
        ptx_loss = outputs.loss * self.args.ptx_coef
        
        if ptx_loss.isnan():
            import ipdb; ipdb.set_trace()
        
        self.model.backward(ptx_loss)
        self.model.step()
        
        ptx_loss = get_all_reduce_mean(ptx_loss)
        
        return {
            'train/ptx_loss': ptx_loss.item(),
        }
    
    @torch.no_grad()
    def eval(self):
        if self.train_dataloader is None:
            return {}        
        self.model.eval()
        
        eval_dataloader = tqdm(
            self.train_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
            position=1,
            leave=False,
        )
        samples = []
        import jsonlines
        for batch in eval_dataloader:
            outputs, inputs = self.eval_step(**to_device(batch, self.args.device))
            logprobs, randimg_logprobs = outputs
            input_ids, label_masks = inputs
            
            batch_size = input_ids.size(0) // 2
            for i in range(batch_size):
                better_ids, worse_ids = input_ids[i], input_ids[batch_size + i]
                better_labels, worse_labels = label_masks[i], label_masks[batch_size + i]
                better_logprobs, worse_logprobs = logprobs[i], logprobs[batch_size + i]
                better_randimg_logprobs, worse_randimg_logprobs = randimg_logprobs[i], randimg_logprobs[batch_size + i]
                better_end_index, worse_end_index = better_ids.nonzero()[-1], worse_ids.nonzero()[-1]
                samples.append({
                    'category_id': int(batch['category_ids'][i][0].item()),
                    'better_ids': better_ids[:better_end_index].tolist(),
                    'worse_ids': worse_ids[:worse_end_index].tolist(),
                    'better_labels': better_labels[:better_end_index].tolist(),
                    'worse_labels': worse_labels[:worse_end_index].tolist(),
                    'better_logprobs': better_logprobs[:better_end_index].tolist(),
                    'worse_logprobs': worse_logprobs[:worse_end_index].tolist(),
                    'better_randimg_logprobs': better_randimg_logprobs[:better_end_index].tolist(),
                    'worse_randimg_logprobs': worse_randimg_logprobs[:worse_end_index].tolist(),
                })
                
                model_name = self.args.model_name_or_path.split('/')[-1] if len(self.args.model_name_or_path.split('/')) <= 2 else self.args.model_name_or_path.split('/')[-2]
                with jsonlines.open('{}/{}_eval_result.jsonl'.format(self.args.output_dir, model_name), mode='a') as writer:
                    writer.write_all(samples[-1:])
                
                
        import ipdb; ipdb.set_trace()
    
    def train(self) -> None:
        """Train the model."""
        if self.args.need_eval:
            self.eval()
        
        self.logger.print('***** Running training *****')

        # ==================================================================
        # 步骤 1: 改进并实现正确的恢复训练 (Resume Training) 逻辑
        # ==================================================================
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        
        # 检查是否提供了有效的检查点路径用于恢复
        if self.args.resume_from_ckpt and os.path.isdir(self.args.resume_from_ckpt):
            self.logger.print(f"Resuming from checkpoint: {self.args.resume_from_ckpt}")
            
            # 使用 DeepSpeed 的 API 来加载模型、优化器和学习率调度器的状态
            # load_checkpoint 的第一个参数是 checkpoint 文件夹的父目录，tag 是文件夹名
            # 例如, resume_from_ckpt = '/path/to/checkpoint-500'
            # 那么 load_dir = '/path/to', tag = 'checkpoint-500'
            load_dir, tag = os.path.split(self.args.resume_from_ckpt)
            self.model.load_checkpoint(load_dir, tag=tag)

            # 加载我们自己保存的 trainer 状态 (global_step, epoch 等)
            # *** 已修正：调用类方法 self.load_trainer_state ***
            trainer_state = self.load_trainer_state(self.args.resume_from_ckpt)
            self.global_step = trainer_state.get("global_step", 0)
            epochs_trained = trainer_state.get("epoch", 0)
            
            # 基于恢复的 global_step，计算出在当前 epoch 中已经训练了多少步
            # 这用于在 dataloader 循环中跳过已经处理过的批次
            if len(self.train_dataloader) > 0:
                steps_trained_in_current_epoch = self.global_step % len(self.train_dataloader)

            self.logger.print(f"  Resumed from global_step: {self.global_step}")
            self.logger.print(f"  Resumed from epoch: {epochs_trained}")
            self.logger.print(f"  Continuing training from step {steps_trained_in_current_epoch} in epoch {epochs_trained + 1}")
        else:
            # 如果不恢复，则从头开始
            self.global_step = 0

        # 计算总训练步数，并初始化进度条，使其从恢复的步数开始
        total_train_steps = self.args.num_train_epochs * len(self.train_dataloader)
        progress_bar = tqdm(
            total=total_train_steps,
            initial=self.global_step, # 让进度条从 global_step 开始
            desc=f'Training {epochs_trained + 1}/{self.args.num_train_epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        # ==================================================================
        # 步骤 2: 主训练循环
        # ==================================================================
        num_prompt_only_batches = len(self.train_dataloader)
        num_ptx_batches = len(self.ptx_train_dataloader)
        num_ptx_replicas = (num_prompt_only_batches + num_ptx_batches - 1) // num_ptx_batches
        
        # 从上次中断的 epoch 开始循环
        for epoch in range(epochs_trained, self.args.num_train_epochs):
            self.model.train()
            
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            for step, (batch, ptx_batch) in enumerate(zip(
                self.train_dataloader,
                itertools.chain.from_iterable([self.ptx_train_dataloader] * num_ptx_replicas),
            )):
                
                if step < steps_trained_in_current_epoch:
                    continue

                info = self.train_step(**to_device(batch, self.args.device))
                if self.use_ptx:
                    ptx_info = self.ptx_step(**to_device(ptx_batch, self.args.device))
                
                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.num_train_epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)
                if self.use_ptx:
                    self.logger.log(ptx_info, step=self.global_step)
                
                if self.global_step > 0 and self.global_step % self.args.save_steps == 0:
                    self.logger.print(f'Saving training checkpoint at step {self.global_step} ...')
                    self.save_training_checkpoint(global_steps=self.global_step, epoch=epoch)
                    self.logger.print('Training checkpoint saved.')
                
                if (
                    self.args.need_eval
                    and self.args.eval_strategy == 'steps'
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)
            
            steps_trained_in_current_epoch = 0

            # self.save_training_checkpoint(global_steps=self.global_step, epoch=epoch)
            # self.save_inference_model(output_dir=os.path.join(self.args.output_dir, "final_model"))
            inference_model_dir = os.path.join(self.args.output_dir, f"inference_model_step_{self.global_step}")
            self.save_inference_model(output_dir=inference_model_dir)
            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.num_train_epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            if hasattr(self.model, 'tput_timer'):
                self.model.tput_timer.update_epoch_count()
        
        # self.logger.print("Training finished. Saving final model for inference...")
        # self.save_inference_model(output_dir=os.path.join(self.args.output_dir, "final_model"))
        # self.logger.print("Final inference model saved in 'final_model' directory.")
    
    # ======================================================================
    # 新增的辅助方法，已整合到类中
    # ======================================================================
    def save_trainer_state(self, output_dir: str, state: dict):
        """Saves custom trainer state to a json file."""
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "custom_trainer_state.json"), "w") as f:
                import json
                json.dump(state, f, indent=4)

    def load_trainer_state(self, output_dir: str) -> dict:
        """Loads custom trainer state from a json file."""
        state_path = os.path.join(output_dir, "custom_trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                import json
                return json.load(f)
        return {}
    
    def save_training_checkpoint(self, global_steps: int, epoch: int) -> None:
        """
        Saves a full training checkpoint that can be used to RESUME training.
        This uses DeepSpeed's checkpointing which saves model, optimizer, and lr_scheduler states.
        """
        dist.barrier()
        
        checkpoint_tag = f'checkpoint-{global_steps}'
        
        # DeepSpeed `save_checkpoint` 会在 `self.args.output_dir` 下创建 `checkpoint_tag` 目录
        self.model.save_checkpoint(self.args.output_dir, tag=checkpoint_tag)
        
        # 保存自定义的 trainer 状态
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_tag)
        trainer_state = {
            "global_step": global_steps,
            "epoch": epoch,
        }
        # *** 已修正：调用类方法 self.save_trainer_state ***
        self.save_trainer_state(checkpoint_dir, trainer_state)
        
        # 保存 tokenizer，方便检查点自包含
        if is_main_process():
            self.tokenizer.save_pretrained(checkpoint_dir)

        dist.barrier()

    def get_non_lora_trainable_weights(self, model: torch.nn.Module) -> dict:
        """
        从模型中提取非LoRA但可训练的权重，主要是 LLaVA 的 mm_projector。
        """
        state_dict = model.state_dict()
        non_lora_trainables = {}
        
        # LLaVA 的投影层通常命名中包含 'mm_projector'
        # 我们也需要保存更新过的 input/output embeddings (如果添加了新token)
        keys_to_match = ['mm_projector', 'embed_tokens', 'lm_head']
        
        for name, param in state_dict.items():
            # 检查参数名是否匹配任何一个关键词
            if any(key in name for key in keys_to_match):
                # 另外，我们需要确保这个参数不是LoRA的一部分
                # LoRA参数通常包含 'lora_A' 或 'lora_B'
                if 'lora_A' not in name and 'lora_B' not in name:
                    non_lora_trainables[name] = param.cpu() # 保存到cpu以避免GPU内存问题
                    
        return non_lora_trainables

    def save_inference_model(
        self,
        output_dir: str,
        model: Optional[torch.nn.Module] = None # 使用 torch.nn.Module 以兼容 deepspeed
    ) -> None:
        """
        Saves a model in Hugging Face format, suitable for INFERENCE.
        For DeepSpeed ZeRO Stage 3, this involves converting the sharded checkpoint.
        """
        dist.barrier()
        
        if model is None:
            model = self.model
        
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
        dist.barrier()

        self.logger.print(f'Saving inference-ready model to "{output_dir}" ...')

        model_to_save: PreTrainedModel = getattr(model, 'module', model)
        is_peft_model = isinstance(model_to_save, PeftModel)
        ds_config = self.ds_train_config
        
        # 保存 Tokenizer 和 Config
        if is_main_process():
            self.tokenizer.save_pretrained(output_dir)
            # 对于PEFT模型，保存基础模型的config
            config_to_save = model_to_save.get_base_model().config if is_peft_model else model_to_save.config
            config_to_save.to_json_file(os.path.join(output_dir, CONFIG_NAME))

        # 保存权重
        if is_peft_model:
            self.logger.print('Saving LoRA adapter weights for inference...')
            if is_main_process():
                # 1. 保存 LoRA 适配器 (adapter_model.safetensors)
                model_to_save.save_pretrained(output_dir)

                # 2. 【关键新增部分】保存 non-lora-trainables (mm_projector)
                self.logger.print('Saving non-LoRA trainable weights (e.g., mm_projector)...')
                
                # 注意：如果使用了ZeRO-3，权重是分片的，直接从 model_to_save 获取可能不完整。
                # LLaVA官方实现使用了一个 get_mm_adapter_state_maybe_zero_3 的复杂函数来处理。
                # 这里我们先用一个简化版本，假设在主进程上可以拿到完整的state_dict。
                # 如果你使用ZeRO-3，这部分可能需要更复杂的处理来聚合分片权重。
                non_lora_weights = self.get_non_lora_trainable_weights(model_to_save)
                
                if non_lora_weights:
                    torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_trainables.bin"))
                else:
                    self.logger.warning("No non-LoRA trainable weights (like mm_projector) were found to save.")
                
        else:
            # 这部分保持不变，用于处理全量微调
            self.logger.print('Saving full model weights for inference...')
            # ... (你的全量保存逻辑)

        dist.barrier()
        self.logger.print('Inference-ready model saved!')