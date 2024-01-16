import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.distributed.fsdp.wrap import wrap

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    logging,
)

from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    repeat_kv,
    apply_rotary_pos_emb,
    LlamaRMSNorm,
    LlamaPreTrainedModel,
    _make_causal_mask,
    _expand_mask,
    LLAMA_INPUTS_DOCSTRING,
    LLAMA_START_DOCSTRING,
)

from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "ZebraConfig"


def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    if x.shape[dim] % block_len == 0:
        return x
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = torch.nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _pad_to_multiple_2d(x: torch.Tensor, block_len: int, dim_1: int, dim_2: int, pad_value: int = 0) -> torch.Tensor:
    pad_len_1 = -x.shape[dim_1] % block_len
    pad_len_2 = -x.shape[dim_2] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim_1] += pad_len_1
        new_shape[dim_2] += pad_len_2
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim_1] = (0, pad_len_1)
    pad[dim_2] = (0, pad_len_2)
    pad = sum(pad[::-1], ())
    x = torch.nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    assert x.shape[
               dim] % block_len == 0, f"sequence length({x.shape[dim]}) should be multiple of block length({block_len})"
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1):]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


def _concatenate_2_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.
    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 0)
    pad = sum(pad[::-1], ())
    # [batch_size, num_blocks, block_len] -> [batch_size, 1 + num_blocks , block_len]
    x = torch.nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list = []
    for i in range(2):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 2 * block_len, ...]
    return torch.cat(blocks_list, dim=sequence_dim)


def _get_local_casual_attention_mask(block_len: int, device=None) -> torch.Tensor:
    m = torch.cat([torch.zeros((block_len, block_len + 1)), torch.ones((block_len, block_len))], dim=-1).to(device)
    m = m.reshape(-1)[: block_len * block_len * 2]
    return m.reshape(block_len, block_len * 2).unsqueeze(0).unsqueeze(0) > 0.5


def _get_local_attention_mask(m: torch.Tensor, block_len: int) -> torch.Tensor:
    """ Construct the local attention mask from the original attention mask.
        The Input shape is: [batch_size, 1, seq_len, seq_len]
        The Output shape is: [batch_size * num_blocks, 1, block_len, 2 * block_len]
    """
    # First Padding to Multiple of block_len
    if m.shape[-2] % block_len != 0 or m.shape[-1] % block_len != 0:
        m = _pad_to_multiple_2d(m, block_len, dim_1=-2, dim_2=-1, pad_value=1)

    # Reshape to [batch_size, 1, num_blocks, block_len, num_blocks, block_len]
    num_blocks = m.shape[-2] // block_len
    output_shape = m.shape[:-2] + (num_blocks, block_len) + (num_blocks, block_len)
    blocked_m = m.reshape(output_shape)

    # Padding One Block at dim -2
    pad = [(0, 0)] * blocked_m.ndim
    pad[-2] = (1, 0)
    pad = sum(pad[::-1], ())
    # [batch_size, 1, num_blocks, block_len, 1 + num_blocks, block_len]
    padded_m = torch.nn.functional.pad(blocked_m, pad=pad, mode="constant", value=1)
    mask_block_list = []
    for i in range(2):
        indices = [slice(0, None)] * padded_m.ndim
        indices[-2] = slice(i, i + num_blocks)
        indices = tuple(indices)
        mask_block_list.append(padded_m[indices])
    # shape of [batch_size, 1, num_blocks, block_len, num_block, 2 * block_len]
    cat_m = torch.cat(mask_block_list, dim=-1)
    # shape of [num_blocks, batch_size, 1, block_len, 2 * block_len]
    ret_m = cat_m[:, :, torch.arange(num_blocks), :, torch.arange(num_blocks), :].transpose(0, 1).transpose(1, 2)
    return ret_m


def attention_mask_func(attn_score, attn_mask):
    dtype = attn_score.dtype
    attn_score = attn_score.mask_fill(attn_mask, torch.finfo(dtype).min)
    # attn_score = attn_score.mask_fill(attn_mask, -10000.0)
    return attn_score


class MaskedSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_func = attention_mask_func

    def forward(self, input, mask):
        dtype = input.dtype
        input = input.to(dtype=torch.float32)
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output).to(dtype)
        return probs


class ZebraMixAttention(nn.Module):
    """Sparse attention implementation by Kaiqiang"""

    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings
        # Added for Mix Attention by Kaiqiang
        self.self_attn_type = config.self_attn_type
        self.block_len = config.window_size
        self.layer_group_size = config.layer_group_size
        self.softmax_func = MaskedSoftmax()

        # Addef for Mix Attention by Kaiqiang
        if self.self_attn_type == "mix":
            if self.layer_id % self.layer_group_size == 0:
                self.self_attn_type = "full"
            else:
                self.self_attn_type = "sparse"

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            rope_theta = self.config.rope_theta
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=rope_theta,
                    scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=rope_theta,
                    scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def full_attention(self, query_states, key_states, value_states, attention_mask, help_args):
        bsz = help_args["bsz"]
        q_len = help_args["q_len"]
        kv_seq_len = help_args["kv_seq_len"]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if self.self_attn_type == "full" and attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Full Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if self.self_attn_type == "full" and attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Full Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        return attn_output, attn_weights

    def sparse_attention(self, query_states, key_states, value_states, attention_mask, help_args):
        """ states: bsz, self.num_heads, q/k/v_len, self.head_dim
        """
        bsz = help_args["bsz"]
        q_len = help_args["q_len"]
        kv_seq_len = help_args["kv_seq_len"]

        assert q_len == kv_seq_len, \
            f"sparse attention only used for training when q_len({q_len} == kv_seq_len({kv_seq_len}))"

        # Convert Attention to 0 (valid) and 1 (invalid)
        if attention_mask is not None:
            attention_mask = (attention_mask < 0)

        # Transpose to shape: bsz, seq_len, num_heads, head_dim
        query_layer = query_states.transpose(1, 2)
        key_layer = key_states.transpose(1, 2)
        value_layer = value_states.transpose(1, 2)

        # Padded to multiple
        query_layer = _pad_to_multiple(query_layer, self.block_len, dim=1, pad_value=0)
        key_layer = _pad_to_multiple(key_layer, self.block_len, dim=1, pad_value=0)
        value_layer = _pad_to_multiple(value_layer, self.block_len, dim=1, pad_value=0)

        padded_seq_len = query_layer.shape[1]
        num_blocks = padded_seq_len // self.block_len

        ###############################################
        # Processing Q,K,V for local attention
        ###############################################

        # split into blocks -> (batch_size, num_blocks, block_len, num_heads_per_partition, dim_per_head)
        query_layer_local = _split_into_blocks(query_layer, self.block_len, dim=1)
        key_layer_local = _split_into_blocks(key_layer, self.block_len, dim=1)
        value_layer_local = _split_into_blocks(value_layer, self.block_len, dim=1)

        # Concatenate 2 blocks for keys and values
        # -> (batch_size, num_blocks, 2 * block_len, num_heads_per_partition, dim_per_head)
        key_layer_local = _concatenate_2_blocks(key_layer_local, block_dim=1, sequence_dim=2)
        value_layer_local = _concatenate_2_blocks(value_layer_local, block_dim=1, sequence_dim=2)

        ###############################################
        # Calculate Local Attention Score
        ###############################################

        # Compute Local Attention Scores
        # -> (batch_size, num_heads_per_partition, num_blocks, block_len, 2 * block_len)
        attn_score_local = torch.einsum(
            "...qhd,...khd->...hqk", query_layer_local, key_layer_local
        ).transpose(1, 2)

        alpha = 1.0 / self.norm_factor
        attn_score_local = alpha * attn_score_local

        # Convert Shape to [b, np, sq, sk] Style
        # -> (batch_size, num_heads_per_partition, padded_seq_len, 2 * block_len)
        new_shape = (bsz, self.num_heads, padded_seq_len, 2 * self.block_len)
        attn_score_local = attn_score_local.reshape(new_shape)

        ###############################################
        # Building Local Attention Masks
        ###############################################

        # Get local attention mask
        # -> (batch_size * num_blocks, 1, block_len, 2 * block_len)
        attn_mask_local = _get_local_attention_mask(attention_mask, self.block_len)
        attn_mask_local_ = _get_local_casual_attention_mask(self.block_len, device=attn_mask_local.device)
        attn_mask_local = torch.logical_or(attn_mask_local, attn_mask_local_)

        # Convert Shape to [b, np, sq, sk] Style
        # -> (batch_size, 1, padded_seq_len, 2 * block_len)
        new_shape = (bsz, 1, padded_seq_len, 2 * self.block_len)
        attn_mask_local = attn_mask_local.reshape(new_shape)

        ###############################################
        # Calculating attention probabilities
        ###############################################

        # using softmax to calculate the attention probabilities
        attn_probs = self.softmax_func(attn_score_local, attn_mask_local)

        # Convert attn_probs
        # -> (batch_size, num_heads_per_partition, num_blocks, block_len, 2 * block_len)
        shape = (bsz, self.num_heads, num_blocks, self.block_len, 2 * self.block_len)
        attn_probs = attn_probs.reshape(shape)
        # Convert attn_probs
        # -> (batch_size, num_blocks, num_heads_per_partition, block_len, 2 * block_len)
        attn_probs = attn_probs.transpose(1, 2)

        # shape: (batch_Size, num_blocks, block_len, n_head, dim_per_head)
        attn_outputs = torch.einsum(
            "...hqk,...khd->...qhd", attn_probs, value_layer_local
        )

        # convert attn_output
        # -> (batch_size, num_blocks * block_len, n_head * dim_per_head)
        attn_outputs = attn_outputs.reshape(
            bsz,
            padded_seq_len,
            self.num_heads * self.head_dim
        )

        # Removing the padded length and transpose
        # -> (batch_size, seq_len, dim_per_partition)
        attn_outputs = attn_outputs.narrow(1, 0, q_len)

        if attn_outputs.size() != (bsz, q_len, self.num_heads * self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads * self.head_dim)}, but is"
                f" {attn_outputs.size()}"
            )

        return attn_outputs, attn_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if position_ids is not None:
            position_ids = position_ids.to(hidden_states.device)
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = query_states.to(hidden_states.device)
        key_states = key_states.to(hidden_states.device)
        value_states = value_states.to(hidden_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # make on same device
        cos = cos.to(hidden_states.device)
        sin = sin.to(hidden_states.device)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        help_args = {
            "bsz": bsz,
            "q_len": q_len,
            "kv_seq_len": kv_seq_len
        }

        attention_mask = attention_mask.to(hidden_states.device)

        if self.self_attn_type == "full":
            # use fully attention for full attention
            attn_output, attn_weights = self.full_attention(
                query_states, key_states, value_states, attention_mask, help_args
            )
        elif use_cache:
            # use full attention with truncated key/value cache for sparse attention
            truncate_attention_mask = attention_mask
            if attention_mask.shape[3] > self.block_len + 1:
                truncate_attention_mask[:, :, :, :-(self.block_len + 1)].fill_(torch.finfo(attention_mask.dtype).min)
            attn_output, attn_weights = self.full_attention(
                query_states, key_states, value_states, truncate_attention_mask, help_args
            )
        else:
            # use sparse attention only for training
            # print("In cache", position_ids, self.layer_id, "use sparse attn")
            attn_output, attn_weights = self.sparse_attention(
                query_states, key_states, value_states, attention_mask, help_args
            )

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        device = x.device
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            t1 = self.act_fn(self.gate_proj(x.to(self.gate_proj.weight.device))).to(device)
            t2 = self.up_proj(x.to(self.up_proj.weight.device)).to(device)
            down_proj = self.down_proj((t1 * t2).to(self.down_proj.weight.device)).to(device)

        return down_proj


class ZebraDecoderLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.self_attn = ZebraMixAttention(layer_id=layer_id, config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.to(residual.device)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class ZebraModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.use_fsdp = config.use_fsdp
        if self.use_fsdp:
            self.layers = nn.ModuleList([
                wrap(ZebraDecoderLayer(layer_id, config)) for layer_id in range(config.num_hidden_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                ZebraDecoderLayer(layer_id, config) for layer_id in range(config.num_hidden_layers)
            ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ZebraForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = ZebraModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
