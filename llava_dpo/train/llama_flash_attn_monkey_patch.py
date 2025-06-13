from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # if output_attentions:
    #     warnings.warn(
    #         "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
    #     )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

# ==================== 主要修改在这里 ====================
        
 # --------------------------------------------------------------------------
    # 2. 精准逻辑判断 (决定走哪条路径)
    # --------------------------------------------------------------------------
    target_layer_for_grad = getattr(self.config, 'standard_attention_layer_idx', None)

    use_standard_path = (
        output_attentions and
        target_layer_for_grad is not None and 
        hasattr(self, 'layer_idx') and # 安全检查，确保 layer_idx 已被注入
        self.layer_idx == target_layer_for_grad
    )

    # --------------------------------------------------------------------------
    # 3. 分支 A: 标准注意力路径 (为目标层执行)
    # --------------------------------------------------------------------------
    if use_standard_path:
        # 1. 计算原始注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 2. 构建基础的2D因果掩码 (causal mask)
        causal_mask = torch.full(
            (q_len, kv_seq_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device
        )
        mask_cond = torch.arange(causal_mask.size(-1), device=attn_weights.device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
        
        # 3. (关键修复) 将2D因果掩码扩展为4D
        #    形状从 (q_len, kv_seq_len) -> (1, 1, q_len, kv_seq_len)
        #    这样它就能和 (bsz, 1, q_len, kv_seq_len) 的 attention_mask 进行广播相加
        causal_mask = causal_mask[None, None, :, :].to(attn_weights.dtype)

        # 4. 如果存在填充掩码 (attention_mask)，则将其与4D因果掩码合并
        # 4. 如果存在填充掩码 (attention_mask)，则将其与4D因果掩码合并
        if attention_mask is not None:
            # attention_mask 传入时通常是 2D 的 [bsz, seq_len]
            # 我们需要将其扩展为 4D 的 [bsz, 1, 1, seq_len] 以便和 attn_weights [bsz, heads, q_len, seq_len] 广播
            # 这是与因果掩码 [1, 1, q_len, seq_len] 相加的关键步骤
            
            # <<< 核心修复：在这里扩展 attention_mask 的维度 >>>
            # 检查 attention_mask 是否为预期的 4D 形状，如果不是 (通常是 2D), 就进行扩展
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]

            # PyTorch的广播机制会自动处理 [1, 1, q, k] + [bsz, 1, 1, k] -> [bsz, 1, q, k]
            causal_mask = causal_mask + attention_mask
        
        # 5. 将最终的掩码应用到注意力分数上
        attn_weights = attn_weights + causal_mask
        
        # 5. 将最终的掩码应用到注意力分数上
        attn_weights = attn_weights + causal_mask
        
        # 6. 计算 Softmax 和最终输出
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 7. 整理形状并进行输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # 8. 返回输出和可微分的注意力权重
        return attn_output, attn_weights, past_key_value

    # Transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        max_s = q_len
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = output.view(bsz, q_len, -1)
    else:
        qkv = qkv.reshape(bsz, q_len, -1)
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

        # output = self.o_proj(output)

    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward


'''
保守版本：
if output_attentions:
    # 1. 计算原始注意力分数
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # 2. 【关键修复】构建正确的4D注意力掩码
    #    不要直接使用传入的 attention_mask，因为它格式不对
    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
        
    # 构建因果掩码 (causal mask)，这是自回归模型所必需的
    # 创建一个下三角矩阵，上三角部分将被屏蔽
    mask = torch.full((q_len, kv_seq_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.dtype)

    # 如果有传入的 padding mask (attention_mask)，则将其合并进来
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            # 假设传入的 attention_mask 是 2D 的 (bsz, kv_seq_len)
            # 将其扩展为 4D 以便和 attn_weights 广播相加
            # 注意：这里我们假设 1 是有效位，0 是屏蔽位
            expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, q_len, kv_seq_len).to(torch.bool)
            # 在需要屏蔽的地方（值为0）填充一个很大的负数
            mask = mask.masked_fill(expanded_mask == 0, torch.finfo(attn_weights.dtype).min)

    # 3. 应用构建好的掩码
    attn_weights = attn_weights + mask

    # 4. Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # 5. 计算输出
    attn_output = torch.matmul(attn_weights, value_states)

    # 6. 整理形状并投影
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    
    return attn_output, attn_weights, past_key_value

# Transform the data into the format required by flash attention
qkv = torch.stack([query_states, key_states, value_states], dim=2)
qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
key_padding_mask = attention_mask

if key_padding_mask is None:
    qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
    cu_q_lens = torch.arange(
        0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
    )
    max_s = q_len
    output = flash_attn_unpadded_qkvpacked_func(
        qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    output = output.view(bsz, q_len, -1)
else:
    qkv = qkv.reshape(bsz, q_len, -1)
    qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
    qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
    output_unpad = flash_attn_unpadded_qkvpacked_func(
        qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
    output = pad_input(output_unpad, indices, bsz, q_len)

    # output = self.o_proj(output)

return self.o_proj(output), None, past_key_value

'''
