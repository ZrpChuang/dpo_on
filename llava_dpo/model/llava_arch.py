#    Copyright 2023 Haotian Liu
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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava_dpo.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        try:
            image_features = self.get_model().mm_projector(image_features)
        except:
            image_features = self.get_model().mm_projector(image_features.bfloat16())
        return image_features

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images
        ):
            vision_tower = self.get_vision_tower()
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                    attention_mask = torch.cat((attention_mask, torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                return input_ids, position_ids, attention_mask, past_key_values, None, labels


            if type(images) is list or images.ndim == 5:
                # --- [多图分支] ---
                # 1. 展平所有图像到一个大的批处理张量
                #    无论输入是 list 还是 5D Tensor，这里都将其统一
                concat_images = torch.cat([image for image in images], dim=0)
                
                # 3. [修复] 强制创建一个内存连续的副本
                images_to_encode = concat_images.contiguous()
                # 5. 一次性编码所有图像
                image_features_encoded = self.encode_images(images_to_encode)
                
                # 6. 根据原始结构拆分回每个样本
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features_encoded, split_sizes, dim=0)
                
            else:
                # 2. [无修改] 保持原始的编码调用
                image_features_encoded = self.encode_images(images)
                
                # 3. [无修改] 保持原始的后续处理
                #    为了统一格式，将 (B, P, D) 的输出拆分成 B 个 (1, P, D) 的列表
                image_features = [feature.unsqueeze(0) for feature in image_features_encoded]
            # TODO: image start / end is not implemented here to support pretraining.
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                raise NotImplementedError

            # Let's just add dummy tensors if they do not exist
            _labels = labels
            _position_ids = position_ids
            _attention_mask = attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            # remove the padding using attention_mask
            input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
            labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

            new_input_embeds = []
            new_labels = []
            for batch_idx, cur_input_ids in enumerate(input_ids):
                # 添加一个总的诊断头
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

                # Retrieve all image features for the current sample
                cur_sample_image_features = image_features[batch_idx]
                

                # Robustness check
                if num_images != cur_sample_image_features.shape[0]:
                    raise ValueError(
                        f"Sample {batch_idx} has {num_images} image tokens, but "
                        f"{cur_sample_image_features.shape[0]} images were provided."
                    )

                if num_images == 0:
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    new_input_embeds.append(cur_input_embeds_1)
                    new_labels.append(labels[batch_idx])
                    continue

                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                    
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

                cur_new_input_embeds = []
                cur_new_labels = []
                cur_sample_image_idx = 0
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        # Insert the correct image feature using an inner-sample index
                        image_feature_to_insert = cur_sample_image_features[cur_sample_image_idx]
                        cur_new_input_embeds.append(image_feature_to_insert)
                        cur_new_labels.append(torch.full((image_feature_to_insert.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_sample_image_idx += 1

                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)

            # 尝试从模型配置中获取最大长度限制
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)

            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
                new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            # Combine them and pad
            max_len = max(x.shape[0] for x in new_input_embeds)
            batch_size = len(new_input_embeds)

            new_input_embeds_padded = []
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len = cur_new_embed.shape[0]
                if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                    new_input_embeds_padded.append(torch.cat((
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        cur_new_embed
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask[i, -cur_len:] = True
                        position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    new_input_embeds_padded.append(torch.cat((
                        cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask[i, :cur_len] = True
                        position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

            if _labels is None:
                new_labels = None
            else:
                new_labels = new_labels_padded

            if _attention_mask is None:
                attention_mask = None
            else:
                attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

            if _position_ids is None:
                position_ids = None

            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False