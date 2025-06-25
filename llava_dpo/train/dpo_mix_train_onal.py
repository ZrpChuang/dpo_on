# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from llava_dpo.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Union

import torch

import transformers

from llava_dpo.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava_dpo.train.llava_trainer_onal_mix import DPOLLaVATrainer
#这里开始改了训练trainer
from llava_dpo import conversation as conversation_lib
from llava_dpo.model import *
from llava_dpo.mm_utils import tokenizer_image_token
from llava_dpo.ds_configs.deepspeed_config import get_deepspeed_eval_config

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="liuhaotian/llava-v1.5-7b")
    ref_model_name_or_path: Optional[str] = field(default="liuhaotian/llava-v1.5-7b")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    # mm_use_im_start_end: bool = field(default=False)

@dataclass
class PTXDataArguments:
    ptx_data_path: str = field(default=None,
                               metadata={"help": "Path to the training data."})
    ptx_lazy_preprocess: bool = False
    ptx_is_multimodal: bool = True
    ptx_image_folder: Optional[str] = field(default=None)
    ptx_image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    '''
    from transformers import TrainingArguments
        help(TrainingArguments)
        查看 TrainingArguments 的帮助文档
    '''
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    
    tune_visual_abstractor: bool = field(default=True)
    freeze_vision_model: bool = field(default=True)
    
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    # <--- MODIFICATION START ---
    # 在这里添加 lambda_cal
    lambda_cal: float = field(
        default=0.1, 
        metadata={"help": "The coefficient for the Reference-Aware Contextual Alignment Loss (RA-CAL)."}
    )
    gamma: float = 1    # CFG
    use_logits: bool = False   # normalized prob for CFG
    dynamic_llm: bool = False   # whether to use the updated p(y|x)
    importance_sampling: Optional[str] = None
    need_eval: bool = False
    ptx_coef: float = 0.1
    instruct_coef: float = 1.0
    region_coef: float = 1.0
    vqa_coef: float = 1.0
    scale_coeff: float = .01    # beta
    label_smoothing: float = 0  # conservative (default)
    not_dynamic: bool = False   # static cDPO
    dynamic_label_smoothing: bool = False   # dynamic cDPO
    language_bias_reduce: bool = False      # p(y|v,x) > p(y|x)
    resume_from_ckpt: Optional[str] = None
    ipo: bool = False
    dynamic_loss_weighting: bool = field(
        default=False,
        metadata={"help": "Whether to use dynamic loss weighting based on attention entropy. "
                          "If False, uses a fixed 0.5/0.5 weight for MK and VCD losses. "
                          "If True, weights are calculated dynamically."}
    )
    lora_enable: bool = False
    lora_r: int = 64   #原本64，16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    save_only_model=True #保存lora检查点
    max_steps=50
    mm_projector_lr: Optional[float] = None
    
    visual_abstractor_lr: Optional[float] = None
    
    group_by_modality_length: bool = field(default=False)
    log_project: Optional[str] = None
    per_device_ptx_train_batch_size: int = 128
    n_random_images: int = 8


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[Sequence[Dict]],  # 明确标注 sources 是一个 "字典列表" 的序列
    data_args: Union[DataArguments, PTXDataArguments]
) -> Sequence[Sequence[Dict]]:
    try:
        is_multimodal = data_args.is_multimodal
    except AttributeError: # 使用更具体的异常捕获
        is_multimodal = data_args.ptx_is_multimodal
    if not is_multimodal:
        return sources

    # 遍历每个对话 (source)
    for source in sources:
        # 遍历该对话中的每一轮 (sentence)
        for sentence in source:
            # 确保 sentence 是一个字典
            if not isinstance(sentence, dict):
                # 如果不是字典，可能数据结构有问题，打印警告并跳过
                rank0_print(f"Warning: Expected a dict for a sentence, but got {type(sentence)}. Skipping.")
                continue

            # 检查 'value' 键是否存在且不为空
            if 'value' in sentence and sentence['value']:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    # 重新组织 value，确保 <image> 在最前面并有换行
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    
                    # 添加 mmtag (如果需要)
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                
                # 替换 <image> 占位符为特殊的 image tokens
                replace_token = DEFAULT_IMAGE_TOKEN
                # if data_args.mm_use_im_start_end:
                #     replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_mpt(
    sources_better: Sequence[str],
    sources_worse: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(better_input_ids=better_input_ids, better_labels=better_targets, 
                worse_input_ids=worse_input_ids, worse_labels=worse_targets)


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_contrastive_llama_2(
    sources_better: Sequence[str],
    sources_worse: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations_better, conversations_worse = [], []
    for source_better, source_worse in zip(sources_better, sources_worse):
        if roles[source_better[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source_better = source_better[1:]
        if roles[source_worse[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source_worse = source_worse[1:]

        conv.messages = []
        for j, sentence in enumerate(source_better):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations_better.append(conv.get_prompt())
        conv.messages = []
        for j, sentence in enumerate(source_worse):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations_worse.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        better_input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations_better], dim=0)
        worse_input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations_worse], dim=0)
    else:
        better_input_ids = tokenizer(
            conversations_better,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        worse_input_ids = tokenizer(
            conversations_worse,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    better_targets = better_input_ids.clone()
    worse_targets = worse_input_ids.clone()
#     - `.clone()` 是PyTorch张量(tensor)的一个方法，用于创建张量的完整副本。
#     - 这个操作会创建一个新的张量，它与原始张量 `better_input_ids` 具有相同的值，但在内存中是完全独立的。
#     - 这个新创建的副本被赋值给变量 `better_targets`。

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    
    def _mask_targets(conversation, target):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    
    for better_conversation, better_target, worse_conversation, worse_target in zip(conversations_better, better_targets, conversations_worse, worse_targets):
        _mask_targets(better_conversation, better_target)
        _mask_targets(worse_conversation, worse_target)        

    return dict(better_input_ids=better_input_ids, better_labels=better_targets, 
                worse_input_ids=worse_input_ids, worse_labels=worse_targets)


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # if i > 0:
            #     round_len -= 1
            #     instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                import ipdb; ipdb.set_trace()
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_contrastive_v1(
    sources_better: Sequence[str],
    sources_worse: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations_better, conversations_worse = [], []
    for source_better, source_worse in zip(sources_better, sources_worse):
        if roles[source_better[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source_better = source_better[1:]
        if roles[source_worse[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source_worse = source_worse[1:]

        conv.messages = []
        for j, sentence in enumerate(source_better):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations_better.append(conv.get_prompt())
        conv.messages = []
        for j, sentence in enumerate(source_worse):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations_worse.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        better_input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations_better], dim=0)
        worse_input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations_worse], dim=0)
    else:
        better_input_ids = tokenizer(
            conversations_better,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        worse_input_ids = tokenizer(
            conversations_worse,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    better_targets = better_input_ids.clone()
    worse_targets = worse_input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    
    def _mask_targets(conversation, target):
        '''
        根据对话结构对目标张量进行掩码处理。
        本函数假设对话通过 `conv.sep2` 分为若干轮（round）。
        它会将 target 张量中每一轮指令部分对应的 token 置为 `IGNORE_INDEX`，实现掩码。
        掩码规则是：每一轮中指令部分的 token 都被设置为 `IGNORE_INDEX`。
        同时函数会检查 target 的总长度是否与 tokenizer 计算的长度一致。
        如果长度不一致，则将所有 token 置为 `IGNORE_INDEX` 并打印警告信息。
        :param conversation: 需要处理的对话字符串。
        :param target: 需要掩码的目标张量。
        '''
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i > 0:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    
    for better_conversation, better_target, worse_conversation, worse_target in zip(conversations_better, better_targets, conversations_worse, worse_targets):
        _mask_targets(better_conversation, better_target)
        _mask_targets(worse_conversation, worse_target)

    return dict(better_input_ids=better_input_ids, better_labels=better_targets, 
                worse_input_ids=worse_input_ids, worse_labels=worse_targets)


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_contrastive_plain(
    sources_better: Sequence[str],
    sources_worse: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations_better, conversations_worse = [], []
    for source_better, source_worse in zip(sources_better, sources_worse):
        assert len(source_better) == 2 == len(source_worse)
        assert DEFAULT_IMAGE_TOKEN in source_better[0]['value'] and DEFAULT_IMAGE_TOKEN in source_worse[0]['value']
        source_better[0]['value'] = DEFAULT_IMAGE_TOKEN
        source_worse[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversations_better.append(source_better[0]['value'] + source_better[1]['value'] + conversation_lib.default_conversation.sep)
        conversations_worse.append(source_worse[0]['value'] + source_worse[1]['value'] + conversation_lib.default_conversation.sep)
    # tokenize conversations
    better_input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations_better]
    better_targets = copy.deepcopy(better_input_ids)
    worse_input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations_worse]
    worse_targets = copy.deepcopy(worse_input_ids)
    for better_target, better_source, worse_target, worse_source in zip(better_targets, sources_better, worse_targets, sources_worse):
        better_tokenized_len = len(tokenizer_image_token(better_source[0]['value'], tokenizer))
        better_target[:better_tokenized_len] = IGNORE_INDEX
        worse_tokenized_len = len(tokenizer_image_token(worse_source[0]['value'], tokenizer))
        worse_target[:worse_tokenized_len] = IGNORE_INDEX

    return dict(better_input_ids=better_input_ids, better_labels=better_targets, 
                worse_input_ids=worse_input_ids, worse_labels=worse_targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    给定一个 sources 列表，每个元素是一个对话列表。该函数实现如下转换：
    1. 在每句话开头添加信号 '### '，结尾添加换行符 '\n'；
    2. 将对话内容拼接在一起；
    3. 对拼接后的对话进行分词（tokenize）；
    4. 深拷贝得到 target，并将 human 说的话用 IGNORE_INDEX 进行掩码。
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_contrastive( # 有对比的数据集
    sources_better: Sequence[str],
    sources_worse: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    给定一个 sources 列表，每个元素是一个对话列表。该函数实现如下转换：
    1. 在每句话开头添加信号 '### '，结尾添加换行符 '\n'；
    2. 将对话内容拼接在一起；
    3. 对拼接后的对话进行分词（tokenize）；
    4. 深拷贝得到 target，并将 human 说的话用 IGNORE_INDEX 进行掩码。
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_contrastive_plain(sources_better, sources_worse, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_contrastive_llama_2(sources_better, sources_worse, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        # print("preprocess_contrastive_v1")
        return preprocess_contrastive_v1(sources_better, sources_worse, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources_better, sources_worse, tokenizer)
    # add end signal and concatenate together
    conversations_better, conversations_worse = [], []
    for source_better, source_worse in zip(sources_better, sources_worse):
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversations_better.append(_add_speaker_and_signal(header, source_better))
        conversations_worse.append(_add_speaker_and_signal(header, source_worse))
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        better_input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations_better]
        worse_input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations_worse]
    else:
        better_input_ids = _tokenize_fn(conversations_better, tokenizer)["input_ids"]
        worse_input_ids = _tokenize_fn(conversations_worse, tokenizer)["input_ids"]

    better_targets, worse_targets = copy.deepcopy(better_input_ids), copy.deepcopy(worse_input_ids)
    for better_target, worse_target, better_source, worse_source in zip(better_targets, worse_targets, sources_better, sources_worse):
        if has_image:
            better_tokenized_lens = get_tokenize_len([header] + [s["value"] for s in better_source])
            worse_tokenized_lens = get_tokenize_len([header] + [s["value"] for s in worse_source])
        else:
            better_tokenized_lens = _tokenize_fn([header] + [s["value"] for s in better_source], tokenizer)["input_ids_lens"]
            worse_tokenized_lens = _tokenize_fn([header] + [s["value"] for s in worse_source], tokenizer)["input_ids_lens"]
        _mask_targets(better_target, better_tokenized_lens, [sentence["from"] for sentence in better_source])
        _mask_targets(worse_target, worse_tokenized_lens, [sentence["from"] for sentence in worse_source])

    return dict(better_input_ids=better_input_ids, better_labels=better_targets, 
                worse_input_ids=worse_input_ids, worse_labels=worse_targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.ptx_image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.ptx_image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.ptx_is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

# /data/ruipeng.zhang/V-DPO/RLHF-V-Dataset/train

class LazyContrastiveDataset(LazySupervisedDataset):
    """
    Dataset for contrastive learning.
    Modified to handle 'mk_value' and 'vcd_value' to create two sets of preference pairs.
    """

# 在 LazyContrastiveDataset 类中

    def _reformat_conversations(self, convs_list: List[Dict], value_key: str) -> List[Dict]:
        """
        Helper function to reformat conversations to the expected format for preprocessing functions.
        It takes a conversation list and a key ('mk_value' or 'vcd_value')
        and returns a new list where the 'gpt' turn has a 'value' field.
        """
        reformatted_convs = []
        for turn in convs_list:
            new_turn = turn.copy()
            if new_turn.get('from') == 'gpt':
                if value_key not in new_turn:
                    # 如果数据不规范，缺少某个value，提供一个默认空字符串，避免程序崩溃
                    new_turn['value'] = "" 
                else:
                    # 将指定的 value (mk_value 或 vcd_value) 赋值给 'value'
                    new_turn['value'] = new_turn.pop(value_key)
                
                # 清理掉另一个key，避免混淆
                other_key = 'vcd_value' if value_key == 'mk_value' else 'mk_value'
                new_turn.pop(other_key, None) # 安全地移除

                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                #  移除了错误的 pop('value', None) 这一行！
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

            reformatted_convs.append(new_turn)
        return reformatted_convs

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Batching should be handled by the DataLoader"
        
        source = sources[0]

        
        # 1. 图像处理 (修改后，支持 bbox_vicrop)
        has_image = ('image' in source or 'images' in source)
        image_tensor = None
        if has_image:
            # 假设每个样本只有一个主图像
            image_file = source.get('image') or (source.get('images') and source['images'][0])
            if not image_file:
                raise ValueError("Image file not found in data source.")

            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            try:
                # 加载原始 PIL 图像
                original_pil_image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except Exception as e:
                rank0_print(f"Warning: Failed to load image {image_file}. Using a placeholder. Error: {e}")
                crop_size = processor.crop_size
                original_pil_image = Image.new('RGB', (crop_size['width'], crop_size['height']))

            # 定义一个可复用的预处理函数
            def process_pil_image(pil_img):
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(img_to_expand, background_color):
                        width, height = img_to_expand.size
                        if width == height: return img_to_expand
                        elif width > height:
                            result = Image.new(img_to_expand.mode, (width, width), background_color)
                            result.paste(img_to_expand, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(img_to_expand.mode, (height, height), background_color)
                            result.paste(img_to_expand, ((height - width) // 2, 0))
                            return result
                    
                    pil_img = expand2square(pil_img, tuple(int(x * 255) for x in processor.image_mean))
                
                return processor.preprocess(pil_img, return_tensors='pt')['pixel_values'][0]

            # --- 核心修改逻辑 ---
            images = []

            # 1. 处理第一张图（完整图）
            full_image_tensor = process_pil_image(original_pil_image)
            images.append(full_image_tensor)

            # 2. 处理第二张图（裁剪图或完整图的副本）
            bbox = source.get("bbox_vicrop")
            if bbox and len(bbox) == 4:
                # 如果有 bbox，裁剪并处理
                cropped_pil_image = original_pil_image.crop(tuple(bbox))
                second_image_tensor = process_pil_image(cropped_pil_image)
                images.append(second_image_tensor)
            else:
                # 如果没有 bbox，则复制第一张图的 tensor (维持旧逻辑)
                images.append(full_image_tensor)
            
            # 最终将两张图的 tensor 堆叠起来
            image_tensor = torch.stack(images, dim=0)

        # 2. 准备两组对话数据源
        
        # 准备 MK 偏好对的对话源
        mk_sources_better_conv = self._reformat_conversations(
            copy.deepcopy(source["conversations"]), "mk_value")
        mk_sources_worse_conv = self._reformat_conversations(
            copy.deepcopy(source.get("contrastive_conversations", source["conversations"])), "mk_value")

        # 准备 VCD 偏好对的对话源
        vcd_sources_better_conv = self._reformat_conversations(
            copy.deepcopy(source["conversations"]), "vcd_value")
        vcd_sources_worse_conv = self._reformat_conversations(
            copy.deepcopy(source.get("contrastive_conversations", source["conversations"])), "vcd_value")

        # 3. 分别对两组数据进行预处理
        
        # --- 处理 MK 数据 ---
# 在 LazyContrastiveDataset.__getitem__ 中找到以下代码块并修改

# --- 处理 MK 数据 ---
        # 原来的调用是 [[...]]，现在改为 [...]
        mk_sources_better = preprocess_multimodal([mk_sources_better_conv], self.data_args) if has_image else [mk_sources_better_conv]
        mk_sources_worse = preprocess_multimodal([mk_sources_worse_conv], self.data_args) if has_image else [mk_sources_worse_conv]

        mk_data_dict = preprocess_contrastive(
            mk_sources_better, # 这里传递的 mk_sources_better 现在是 [[{'from':...}]]
            mk_sources_worse,  # 这里传递的 mk_sources_worse 现在是 [[{'from':...}]]
            self.tokenizer,
            has_image=has_image,
        )

        # --- 处理 VCD 数据 ---
        # 同样，原来的调用是 [[...]]，现在改为 [...]
        vcd_sources_better = preprocess_multimodal([vcd_sources_better_conv], self.data_args) if has_image else [vcd_sources_better_conv]
        vcd_sources_worse = preprocess_multimodal([vcd_sources_worse_conv], self.data_args) if has_image else [vcd_sources_worse_conv]

        vcd_data_dict = preprocess_contrastive(
            vcd_sources_better, # vcd_sources_better 现在也是 [[{'from':...}]]
            vcd_sources_worse,  # vcd_sources_worse 现在也是 [[{'from':...}]]
            self.tokenizer,
            has_image=has_image,
        )

        # 4. 组合成最终的返回字典
        data_dict = dict(
            # MK 数据 (原始字段名)
            better_input_ids=mk_data_dict["better_input_ids"][0],
            better_labels=mk_data_dict["better_labels"][0],
            worse_input_ids=mk_data_dict["worse_input_ids"][0],
            worse_labels=mk_data_dict["worse_labels"][0],
            
            # VCD 数据 (新字段名)
            vcd_better_input_ids=vcd_data_dict["better_input_ids"][0],
            vcd_better_labels=vcd_data_dict["better_labels"][0],
            vcd_worse_input_ids=vcd_data_dict["worse_input_ids"][0],
            vcd_worse_labels=vcd_data_dict["worse_labels"][0],
        )

        # 5. 添加图像和其他元数据
        if has_image:
            data_dict['image'] = image_tensor
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(2, 3, crop_size['height'], crop_size['width'])
        
        data_dict['category'] = source.get('category', 'default')
        if 'clip_scores' in source:
            data_dict['confidence'] = source['clip_scores'][0] / (source['clip_scores'][0] + source['clip_scores'][1])
        else:
            data_dict['confidence'] = 0.8
            
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def get_pure_text(input_ids: torch.Tensor, labels: torch.Tensor):
    if IMAGE_TOKEN_INDEX in input_ids:
        img_idx = input_ids.eq(IMAGE_TOKEN_INDEX).nonzero()[0][0].item()
        txt_input_ids = torch.cat([input_ids[:img_idx], input_ids[img_idx+1:]], dim=0)
        txt_labels = torch.cat([labels[:img_idx], labels[img_idx+1:]], dim=0)
    else:
        txt_input_ids, txt_labels = input_ids, labels
    return txt_input_ids, txt_labels


def get_output(input_ids: torch.Tensor, labels: torch.Tensor, bos_token_id: int):
    start_idx = labels.ne(IGNORE_INDEX).nonzero()[0][0].item()
    only_labels = labels[start_idx-1:]
    only_input_ids = torch.cat([torch.Tensor([bos_token_id]), input_ids[start_idx:]], dim=0)
    return only_input_ids, only_labels

@dataclass
class DataCollatorForContrastiveDataset(object):
    """
    Collate examples for contrastive learning.
    Modified to handle both 'mk' and 'vcd' preference pairs.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 解包所有需要的字段，包括新增的 vcd 字段
        keys_to_collate = (
            "better_input_ids", "better_labels", "worse_input_ids", "worse_labels",
            "vcd_better_input_ids", "vcd_better_labels", "vcd_worse_input_ids", "vcd_worse_labels",
            "category", "confidence"
        )
        collated_data = {key: [instance[key] for instance in instances] for key in keys_to_collate}

        # 2. 对 MK 偏好对进行 Padding 和 Chunk (原始逻辑)
        mk_input_ids = collated_data["better_input_ids"] + collated_data["worse_input_ids"]
        mk_labels = collated_data["better_labels"] + collated_data["worse_labels"]
        
        mk_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            mk_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        mk_labels_padded = torch.nn.utils.rnn.pad_sequence(
            mk_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        better_input_ids, worse_input_ids = mk_input_ids_padded.chunk(chunks=2, dim=0)
        better_labels, worse_labels = mk_labels_padded.chunk(chunks=2, dim=0)

        # 3. 对 VCD 偏好对进行 Padding 和 Chunk (新增逻辑)
        vcd_input_ids = collated_data["vcd_better_input_ids"] + collated_data["vcd_worse_input_ids"]
        vcd_labels = collated_data["vcd_better_labels"] + collated_data["vcd_worse_labels"]

        vcd_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            vcd_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        vcd_labels_padded = torch.nn.utils.rnn.pad_sequence(
            vcd_labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        vcd_better_input_ids, vcd_worse_input_ids = vcd_input_ids_padded.chunk(chunks=2, dim=0)
        vcd_better_labels, vcd_worse_labels = vcd_labels_padded.chunk(chunks=2, dim=0)

        # 4. 处理 category 和 confidence (保持不变)
        # 注意：get_pure_text 和 get_output 在你的原始代码中被调用但结果未使用，我遵循了此模式。
        # 如果你的 DPOLLaVATrainer 需要这些，你可能需要为两组数据分别计算它们。
        
        txt2id = {
            'instruct': 0, 'region': 1, 'vqa': 2, 'instructions': 0, 'default': 0,
            'vcr_txt_pair': -1, 'coco_txt_pair': -2, 'sherlock_txt_pair': -3,
            'vcr_img_pair': -4, 'coco_img_pair': -5, 'sherlock_img_pair': -6, 'sherlock_img_pair_inf': -7,
        }
        category_ids = torch.Tensor([txt2id.get(x, 0) for x in collated_data["category"]]).unsqueeze(-1)
        confidence = torch.Tensor(collated_data["confidence"]).unsqueeze(-1)
        
        # 5. 构造最终的 batch 字典
        batch = dict(
            # MK 偏好对
            better_input_ids=better_input_ids[:, :self.tokenizer.model_max_length],
            better_labels=better_labels[:, :self.tokenizer.model_max_length],
            better_attention_mask=better_input_ids.ne(self.tokenizer.pad_token_id)[:, :self.tokenizer.model_max_length],
            worse_input_ids=worse_input_ids[:, :self.tokenizer.model_max_length],
            worse_labels=worse_labels[:, :self.tokenizer.model_max_length],
            worse_attention_mask=worse_input_ids.ne(self.tokenizer.pad_token_id)[:, :self.tokenizer.model_max_length],
            
            # VCD 偏好对
            vcd_better_input_ids=vcd_better_input_ids[:, :self.tokenizer.model_max_length],
            vcd_better_labels=vcd_better_labels[:, :self.tokenizer.model_max_length],
            vcd_better_attention_mask=vcd_better_input_ids.ne(self.tokenizer.pad_token_id)[:, :self.tokenizer.model_max_length],
            vcd_worse_input_ids=vcd_worse_input_ids[:, :self.tokenizer.model_max_length],
            vcd_worse_labels=vcd_worse_labels[:, :self.tokenizer.model_max_length],
            vcd_worse_attention_mask=vcd_worse_input_ids.ne(self.tokenizer.pad_token_id)[:, :self.tokenizer.model_max_length],
            
            # 共享元数据
            category_ids=category_ids,
            confidence=confidence,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_contrastive_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                 data_args) -> Dict:
    """Make dataset and collator for contrastive learning."""
    train_dataset = LazyContrastiveDataset(tokenizer=tokenizer,
                                           data_path=data_args.data_path,
                                           data_args=data_args)
    #检查一下多模态
    data_collator = DataCollatorForContrastiveDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)  #返回一个字典


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.ptx_data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(ptx_train_dataset=train_dataset,
                ptx_eval_dataset=None,
                ptx_data_collator=data_collator)


# =========================================================================
#             最终修复版: dpo_mix_train_onal.py 中的 train() 函数
# =========================================================================


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, PTXDataArguments, TrainingArguments))
    model_args, data_args, ptx_data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            ref_model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.ref_model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    ref_model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    ref_model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        ref_model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        ref_vision_tower = ref_model.get_vision_tower()
        ref_vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        if ptx_data_args.ptx_data_path is not None:
            ptx_data_args.image_processor = vision_tower.image_processor
            ptx_data_args.ptx_is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        ref_model.config.image_aspect_ratio = data_args.image_aspect_ratio
        ref_model.config.tokenizer_padding_side = tokenizer.padding_side
        ref_model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        ref_model.config.tune_mm_mlp_adapter = False

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        ref_model.config.freeze_mm_mlp_adapter = True
        for p in ref_model.get_model().mm_projector.parameters():
            p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        
        ref_model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        ref_model.config.mm_projector_lr = training_args.mm_projector_lr
        ref_model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        ref_model_args = copy.deepcopy(model_args)
        ref_model_args.freeze_mm_mlp_adapter = True
        ref_model_args.tune_mm_mlp_adapter = False
        ref_model.initialize_vision_tokenizer(ref_model_args, tokenizer=tokenizer)
        
        if ptx_data_args.ptx_data_path is not None:
            ptx_data_args.mm_use_im_start_end = data_args.mm_use_im_start_end
            

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_contrastive_data_module(tokenizer=tokenizer, data_args=data_args)  #构造数据集debug模板开始
    if ptx_data_args.ptx_data_path is not None:
        ptx_data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=ptx_data_args)
    else:
        ptx_data_module = dict(ptx_train_dataset=None, ptx_eval_dataset=None, ptx_data_collator=None)
    ds_train_config = training_args.deepspeed_plugin.hf_ds_config.config
    ds_eval_config = get_deepspeed_eval_config(
        stage=ds_train_config['zero_optimization']['stage'],
        offload='optimizer',
        fp16=ds_train_config['fp16']['enabled'],
        bf16=ds_train_config['bf16']['enabled'],
    )
    training_args.model_name_or_path = model_args.model_name_or_path

    # ==================== 在这里执行第一步 (最终正确版 v3) ====================

    # --- 处理主模型 (PEFT 模型) ---
    # 路径: model.base_model.model.model.layers
    print("Injecting layer indices into the main PEFT model...")

    # 获取被PEFT包装最深层的 LlavaLlamaForCausalLM 对象
    deepest_model = model.base_model.model 
    for i, layer in enumerate(deepest_model.model.layers):
        layer.self_attn.layer_idx = i
        layer.self_attn.config = deepest_model.config # config 在这一层
    print("Done.")


    # --- 处理参考模型 (原始模型) ---
    # 路径: ref_model.model.layers
    if ref_model is not None:
        print("Injecting layer indices into the reference model...")
        for i, layer in enumerate(ref_model.model.layers):
            layer.self_attn.layer_idx = i
            layer.self_attn.config = ref_model.config
        print("Done.")

    # ==========================================================


    trainer = DPOLLaVATrainer(
        training_args,
        model, 
        ref_model,
        ds_train_config, 
        ds_eval_config,
        tokenizer=tokenizer,
        **data_module, #字典解开包
        **ptx_data_module
    )

    if False and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    # trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
