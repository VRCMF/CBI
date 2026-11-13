
from typing import Callable, Optional, Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn

import loralib.layers_encoder as loralayer

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, logging
from transformers import Qwen2Config
from loralib.adapter import Adapter, TemporalRelationAdapter

import math


logger = logging.get_logger(__name__)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int, lora_config=None, use_adapter=False, use_lora=False, lora_split=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        self.use_lora = use_lora
        self.use_adapter = use_adapter
        self.lora_split = lora_split
        # ==========================
        # implement the ACL 25 paper --> a span-based approach for clinical temporal relation extraction
        # ipdb.set_trace()
        self.use_routing = lora_config.use_routing if lora_config is not None else False
        # self.use_routing = False
        
        # Query projection
        if use_adapter:
            # Adapter mode: use frozen standard layers
            self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        else:
            # LoRA mode: create query layer based on configuration
            if self.use_lora and self.use_routing:
                if not self.lora_split:
                    # LoRA with routing (no split): use VLLinear
                    self.q_proj = loralayer.VLLinear(
                        config.hidden_size, config.num_attention_heads * self.head_dim,
                        r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.lora_dropout,
                        element_mul=lora_config.element_mul,
                        element_add=lora_config.element_add,
                        element_mul_expand=lora_config.element_mul_expand,
                        routing_xt=lora_config.routing_xt,
                        bias=True
                    )
                else:
                    # LoRA with routing and split: frozen layer + VLLinearSplit
                    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
                    for param in self.q_proj.parameters():
                        param.requires_grad = False
                    self.lora_vl_query = loralayer.VLLinearSplit(
                        config.hidden_size, config.num_attention_heads * self.head_dim,
                        r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.lora_dropout,
                        bias=True
                    )
            elif self.use_lora and not self.use_routing:
                # LoRA without routing: use standard Linear
                self.q_proj = loralayer.Linear(
                    config.hidden_size, config.num_attention_heads * self.head_dim,
                    r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    bias=True
                )
            else:
                # No LoRA: standard layer
                self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        
        # Key is always frozen
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        
        # Value projection
        if use_adapter:
            # Adapter mode: use frozen standard layers
            self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        else:
            # LoRA mode: create value layer based on configuration
            if self.use_lora and self.use_routing:
                if not self.lora_split:
                    # LoRA with routing (no split): use VLLinear
                    self.v_proj = loralayer.VLLinear(
                        config.hidden_size, config.num_key_value_heads * self.head_dim,
                        r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.lora_dropout,
                        element_mul=lora_config.element_mul,
                        element_add=lora_config.element_add,
                        element_mul_expand=lora_config.element_mul_expand,
                        routing_xt=lora_config.routing_xt,
                        bias=True
                    )
                else:
                    # LoRA with routing and split: frozen layer + VLLinearSplit
                    self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
                    for param in self.v_proj.parameters():
                        param.requires_grad = False
                    self.lora_vl_value = loralayer.VLLinearSplit(
                        config.hidden_size, config.num_key_value_heads * self.head_dim,
                        r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.lora_dropout,
                        bias=True
                    )
            elif self.use_lora and not self.use_routing:
                # LoRA without routing: use standard Linear
                self.v_proj = loralayer.Linear(
                    config.hidden_size, config.num_key_value_heads * self.head_dim,
                    r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    bias=True
                )
            else:
                # No LoRA: standard layer
                self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_hidden_states: Optional[torch.Tensor] = None,
        multiply_ones: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Compute query states based on configuration
        if self.use_adapter:
            query_states = self.q_proj(hidden_states)
        elif self.use_lora and self.use_routing:
            if not self.lora_split:
                # VLLinear forward: (x, vis_feat, multiply_ones)
                query_states = self.q_proj(hidden_states, vision_hidden_states, multiply_ones)
            else:
                # Frozen layer + VLLinearSplit
                query_states = self.q_proj(hidden_states)
                query_states_vl = self.lora_vl_query(hidden_states, vision_hidden_states, multiply_ones=multiply_ones)
                query_states = query_states + query_states_vl
        else:
            query_states = self.q_proj(hidden_states)
        
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # Compute value states based on configuration
        if self.use_adapter:
            value_states = self.v_proj(hidden_states)
        elif self.use_lora and self.use_routing:
            if not self.lora_split:
                value_states = self.v_proj(hidden_states, vision_hidden_states, multiply_ones)
            else:
                value_states = self.v_proj(hidden_states)
                value_states_vl = self.lora_vl_value(hidden_states, vision_hidden_states, multiply_ones=multiply_ones)
                value_states = value_states + value_states_vl
        else:
            value_states = self.v_proj(hidden_states)
        
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int, lora_config=None, use_adapter=False, use_lora=False, lora_split=False, freeze_ln=False, adapt_pos="post_module", fusion_method="original", dataset_name=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx, lora_config=lora_config, use_adapter=use_adapter, use_lora=use_lora, lora_split=lora_split)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

        # Adapter support
        self.use_adapter = use_adapter
        self.fusion_method = fusion_method
        if use_adapter and lora_config is not None:
            if fusion_method == "temporal_relation":
                self.adapter = TemporalRelationAdapter(config, lora_config, dataset_name)
            else:  # original fusion method
                self.adapter = Adapter(config, lora_config, dataset_name)
        else:
            # Freeze parameters when not using adapter
            for param in self.mlp.parameters():
                param.requires_grad = False
        
        # Freeze layer norms if requested
        if freeze_ln:
            for param in self.input_layernorm.parameters():
                param.requires_grad = False
            for param in self.post_attention_layernorm.parameters():
                param.requires_grad = False
        
        self.adapt_pos = adapt_pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        vision_hidden_states: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Apply adapter before attention if requested
        if self.use_adapter and self.adapt_pos == "pre_norm":
            hidden_states = self.adapter(hidden_states, vision_hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            vision_hidden_states=vision_hidden_states,
            multiply_ones=False,
            **kwargs,
        )

        # Apply adapter after attention if requested
        if self.use_adapter and self.adapt_pos == "post_module":
            hidden_states = self.adapter(hidden_states, vision_hidden_states)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Apply adapter before MLP if requested
        if self.use_adapter and self.adapt_pos == "pre_norm":
            hidden_states = self.adapter(hidden_states, vision_hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Apply adapter after MLP if requested
        if self.use_adapter and self.adapt_pos == "post_module":
            hidden_states = self.adapter(hidden_states, vision_hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config, lora_config=None, fusion_layer=None, use_adapter=False, use_lora=False, lora_split=False, freeze_ln=False, adapt_pos="post_module", fusion_method="original", dataset_name=None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Create layers with adapter support
        if fusion_layer is None:
            fusion_layer = list(range(config.num_hidden_layers))
            
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                config,
                layer_idx=i,
                lora_config=lora_config,
                use_adapter=use_adapter and i in fusion_layer,
                use_lora=use_lora,
                lora_split=lora_split,
                freeze_ln=freeze_ln,
                adapt_pos=adapt_pos,
                fusion_method=fusion_method,
                dataset_name=dataset_name
            ) for i in range(config.num_hidden_layers)
        ])
        
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_hidden_states: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # # ==========================
        # # implement the ACL 25 paper --> a span-based approach for clinical temporal relation extraction
        # # ipdb.set_trace()
        # hidden_states = hidden_states*vision_hidden_states.expand_as(hidden_states)
        # vision_hidden_states = None
        # # ==========================

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                vision_hidden_states=vision_hidden_states,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config, lora_config=None, fusion_layer=None, use_adapter=False, use_lora=False, lora_split=False, freeze_ln=False, adapt_pos="post_module", fusion_method="original", dataset_name=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(
            config, 
            lora_config=lora_config,
            fusion_layer=fusion_layer,
            use_adapter=use_adapter,
            use_lora=use_lora,
            lora_split=lora_split,
            freeze_ln=freeze_ln,
            adapt_pos=adapt_pos,
            fusion_method=fusion_method,
            dataset_name=dataset_name
        )
        
        # Handle dropout configuration
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            dropout = config.hidden_dropout
        else:
            dropout = 0.1  # Default dropout rate
            
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.use_routing = lora_config.use_routing if lora_config is not None else False

        # Initialize weights and apply final processing
        self.post_init()

        # Freeze parameters if not using adapter
        if not use_adapter:
            for param in self.parameters():
                param.requires_grad = False
            # Unfreeze classification layers
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.dropout.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_hidden_states: Optional[torch.FloatTensor] = None,
        dataset_name: Optional[str] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
        vision_hidden_states (`torch.FloatTensor`, *optional*):
            Vision hidden states for routing if using adapter with routing.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Don't concatenate vision embeddings with text embeddings - let adapter handle them separately
        # This follows Mistral's approach where vision_hidden_states are passed through to the adapter

        transformer_outputs = self.model(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision_hidden_states=vision_hidden_states,
        )

        hidden_states = transformer_outputs[0]
        
        # Get sequence lengths from attention mask
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(-1) - 1
        else:
            sequence_lengths = torch.tensor([hidden_states.shape[1] - 1] * hidden_states.shape[0], 
                                         device=hidden_states.device)
        
        # Get the logits for the last non-masked token in each sequence
        pooled_hidden_states = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]
        pooled_hidden_states = self.dropout(pooled_hidden_states)
        logits = self.classifier(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



__all__ = [
    "Qwen2PreTrainedModel",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
    "Qwen2ForQuestionAnswering",
]
