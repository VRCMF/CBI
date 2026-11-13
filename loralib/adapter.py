import torch
import torch.nn as nn
from transformers.activations import ACT2FN

import ipdb
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid
from typing import Optional, List
import os
import numpy as np



class Adapter(nn.Module):
    # Adapter with all routing functions implemented
    def __init__(self, config, lora_config, dataset_name=None):
        super().__init__()
        self.input_dim = config.hidden_size
        self.down_sample_size = lora_config.r_lora
            
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 
        # self.activation = ACT2FN[config.hidden_activation]
        act = (
            getattr(config, "hidden_activation", None) or
            getattr(config, "hidden_act", None) or
            getattr(config, "activation_function", None) or
            "silu"  # 安全默认
        )
        self.activation = act if callable(act) else ACT2FN[act]


        self.use_routing = lora_config.use_routing
        self.add_relu = lora_config.add_relu
        self.element_mul = lora_config.element_mul
        self.element_add = lora_config.element_add
        self.element_mul_expand = lora_config.element_mul_expand
        self.routing_xt = lora_config.routing_xt
        self.conv_kernel_size = lora_config.conv_kernel_size
        if self.element_mul_expand == True:
            self.down_size = lora_config.r_lora
            self.dw_conv_1 = nn.Conv1d(self.down_size, self.down_size, kernel_size=self.conv_kernel_size, padding=self.conv_kernel_size//2, groups=self.down_size)
            self.dw_conv_2 = nn.Conv1d(self.down_size, self.down_size, kernel_size=self.conv_kernel_size, padding=self.conv_kernel_size//2, groups=self.down_size)
            # self.dw_conv_3 = nn.Conv1d(self.down_size, self.down_size, kernel_size=5, padding=2, groups=self.down_size)

    def forward(self, hidden_states, vision_hidden_states):
        residual = hidden_states
        output_states = self.down_sampler(hidden_states)
        
        if self.use_routing and vision_hidden_states is not None:
            vision_hidden_down = self.down_sampler(vision_hidden_states)
            if self.element_mul:
                # routing function: x_t \circ x'_v
                output_states = output_states * vision_hidden_down
            elif self.element_add:
                # routing function: x_t + x'_v
                output_states = output_states + vision_hidden_down.expand_as(output_states)
            elif self.element_mul_expand:
                # routing function: x_t  x''_v
                # output_states = output_states @ vision_hidden_down.expand(vision_hidden_down.shape[0], self.down_sample_size, self.down_sample_size)
                xt = output_states
                xv = vision_hidden_down
                xt = self.dw_conv_1(xt.transpose(1,2)).transpose(1,2)
                xv = self.dw_conv_2(xv.transpose(1,2)).transpose(1,2)
                z = xt * xv
                # z = self.dw_conv_3(z.transpose(1,2)).transpose(1,2)
                output_states = z
            elif self.routing_xt:
                # routing function: x_t (x_t)^T  x_v, does not work very well, yet not fully explored
                output_states = output_states - vision_hidden_down.expand_as(output_states)
            elif not self.add_relu:
                # routing function: x_t (x_v)^T  x_v
                bsz, vis_len, vis_dim  = vision_hidden_down.shape
                output_states = output_states @ vision_hidden_down.reshape(bsz, vis_dim, vis_len) @ vision_hidden_down
            else:
                # routing function: x_t (x_v)^T  x_v with ReLU
                bsz, vis_len, vis_dim  = vision_hidden_down.shape
                output_states = torch.nn.functional.relu(output_states) @ torch.nn.functional.relu(vision_hidden_down.reshape(bsz, vis_dim, vis_len)) @ vision_hidden_down
        
        output_states = self.activation(output_states)
        output_states = self.up_sampler(output_states)
        output_states = output_states + residual
        return output_states

class TemporalRelationAdapter(nn.Module):
    def __init__(self, config, lora_config, dataset_name=None):
        super().__init__()
        self.input_dim = config.hidden_size
        self.down_size = lora_config.r_lora  # bottleneck dimension (low-rank size)

        # Low-rank projections
        self.down_sampler = nn.Linear(self.input_dim, self.down_size)
        self.up_sampler   = nn.Linear(self.down_size, self.input_dim)
        # Non-linear activation function
        self.activation = ACT2FN[config.hidden_act]
        # # ========
        self.dw_conv_1 = nn.Conv1d(self.down_size, self.down_size, kernel_size=5, padding=2, groups=self.down_size)
        self.dw_conv_2 = nn.Conv1d(self.down_size, self.down_size, kernel_size=5, padding=2, groups=self.down_size)
        self.dw_conv_3 = nn.Conv1d(self.down_size, self.down_size, kernel_size=5, padding=2, groups=self.down_size)

    def forward(self, event1_vec: torch.Tensor, event2_vec: torch.Tensor, context_vec: torch.Tensor = None) -> torch.Tensor:
        # 1. Down-sample both events (and context if provided) to bottleneck dimension
        e1_down = self.down_sampler(event1_vec)  # [batch, down_size]
        e2_down = self.down_sampler(event2_vec)  # [batch, down_size]
        
        xt = e1_down
        xv = e2_down
        # # # ====
        # z = xt * xv
        # # depthwise conv: groups = d_down
        # z = z.transpose(1,2)             # [B, d, L]
        # z = self.dw_conv(z)              # nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d)
        # z = z.transpose(1,2)             # back to [B, L, d]
        # combined_down = z
        # # # ====
        xt = self.dw_conv_1(xt.transpose(1,2)).transpose(1,2)
        xv = self.dw_conv_2(xv.transpose(1,2)).transpose(1,2)
        z = xt * xv
        # z = self.dw_conv_3(z.transpose(1,2)).transpose(1,2)
        combined_down = z

        # # ========
        # Q = self.wq(xt)         # [batch, seq1, r_lora]
        # K = self.wk(xv)         # [batch, seq2, r_lora]
        # V = self.wv(xv)         # [batch, seq2, r_lora]
        # # 缩放点积注意力
        # scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(Q.size(-1))
        # attn = torch.softmax(scores, dim=-1)            # [batch, seq1, seq2]
        # z = torch.matmul(attn, V)
        # combined_down = z
        # ========


        # 5. Apply non-linearity to the combined signal
        combined_down = self.activation(combined_down)

        # 6. Up-sample back to original hidden size
        output_delta = self.up_sampler(combined_down)
        fused_output = event1_vec + output_delta
        return fused_output