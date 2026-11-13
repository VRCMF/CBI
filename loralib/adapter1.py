import torch
import torch.nn as nn
from transformers.activations import ACT2FN

import ipdb
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Optional, List
import os
import numpy as np



# Add missing functions for direct import
def auto_docstring(func):
    """Decorator for auto-docstring (placeholder)"""
    return func

def get_torch_version():
    """Get torch version (placeholder)"""
    import torch
    return torch.__version__


# ============================
event_1_tokens = [['w', 'ledged', 'that', 'they', 'needed', 'to', '[', 'event', '1', ']', 'work', '[/', 'event', '1', ']', 'harder', 'to', 'solve', 'the', 'cases', 'o'], ['out', 'break', 'of', 'holy', 'war', 'could', '[', 'event', '1', ']', 'bring', '[/', 'event', '1', ']', 'thousands', 'of', 'Americans', 'hom'], ['.', '``', 'This', 'convention', 'will', 'be', '[', 'event', '1', ']', 'seen', '[/', 'event', '1', ']', 'as', 'a', 'turning', 'point', 'in', 'our', 'h'], ['r', 'warning', 'that', 'we', 'will', 'be', '[', 'event', '1', ']', 'pre', 'pared', '[/', 'event', '1', ']', 'to', 'act', 'again', 'if', 'we', 'have', 'e'], ['t', 'amina', 'has', 'with', 'stood', 'any', '[', 'event', '1', ']', 'dis', 'rupt', 'ions', '[/', 'event', '1', ']', 'caused', 'so', 'far', 'by', 'the', 'fi'], ['est', 'ion', 'that', 'Mrs', '.', 'Cur', 'rie', 'was', '[', 'event', '1', ']', 'aware', '[/', 'event', '1', ']', 'of', 'any', 'legal', 'or', 'ethical', 'im'], ['e', 'for', 'the', 'November', '1995', '[', 'event', '1', ']', 'ass', 'ass', 'ination', '[/', 'event', '1', ']', 'of', 'R', 'abin', 'at', 'a', 'Tel', 'Aviv'], ['if', 'we', 'have', 'evidence', 'he', 'is', '[', 'event', '1', ']', 't', 'rying', '[/', 'event', '1', ']', 'to', 'rebuild', 'his', 'weapons', 'of'], ['rd', 'an', 'arrived', 'in', 'Washington', '[', 'event', '1', ']', 'seeking', '[/', 'event', '1', ']', 'to', 'med', 'iate', 'the', 'Persian', 'Gu'], ['how', 'the', 'disputes', 'could', 'be', '[', 'event', '1', ']', 'res', 'olved', '[/', 'event', '1', '].', 'Their', 'solution', 'would', 'req'], ['r', 'warning', 'that', 'we', 'will', 'be', '[', 'event', '1', ']', 'pre', 'pared', '[/', 'event', '1', ']', 'to', 'act', 'again', 'if', 'we', 'have', 'e'], ['At', 'al', 'anta', 'and', 'last', 'month', "'s", '[', 'event', '1', ']', 'expl', 'osion', '[/', 'event', '1', ']', 'at', 'an', 'Alabama', 'women', "'s", 'cl'], ['t', 'amina', 'has', 'with', 'stood', 'any', '[', 'event', '1', ']', 'dis', 'rupt', 'ions', '[/', 'event', '1', ']', 'caused', 'so', 'far', 'by', 'the', 'fi'], ['Iraq', ',', 'oil', 'prices', 'have', 'been', '[', 'event', '1', ']', 'fall', 'ing', '[/', 'event', '1', '],', 'that', "'s", 'because', 'of', 'a', 'wor', 'l'], ['ro', 'secut', 'ors', 'said', 'they', 'had', '[', 'event', '1', ']', 'identified', '[/', 'event', '1', ']', 'different', 'suspects', 'in', 'si'], ['f', 'R', 'abin', 'at', 'a', 'Tel', 'Aviv', 'peace', '[', 'event', '1', ']', 'r', 'ally', '[/', 'event', '1', '].', 'Newspaper', 'reports', 'have', 'sa'], ['est', 'ion', 'that', 'Mrs', '.', 'Cur', 'rie', 'was', '[', 'event', '1', ']', 'aware', '[/', 'event', '1', ']', 'of', 'any', 'legal', 'or', 'ethical', 'im'], ['Clinton', 'discussed', 'the', 'Iraq', '[', 'event', '1', ']', 'c', 'risis', '[/', 'event', '1', ']', 'with', 'the', 'one', 'ally', 'who', 'has'], ['ma', 'women', "'s", 'clinic', 'might', 'be', '[', 'event', '1', ']', 'related', '[/', 'event', '1', '].', 'Pierre', 'Thomas', 'has', 'the', 'la'], ['ast', '.', '', 'But', 'now', 'we', "'re", 'going', 'to', '[', 'event', '1', ']', 'go', '[/', 'event', '1', ']', 'back', 'to', 'New', 'York', ',', 'where', 'Dian']]
event_2_tokens = [['killing', 'and', 'other', 'evidence', '[', 'event', '2', ']', 'se', 'em', '[/', 'event', '2', ']', 'to', 'indicate', 'that', 'the', 'eight'], ['de', 'firing', 'across', 'the', 'bow', 'to', '[', 'event', '2', ']', 'h', 'alt', '[/', 'event', '2', ']', 'a', 'ship', '.', 'In', 'the', 'air', ',', 'U', '.', 'S', '.', 'Ai'], ['the', 'thing', 'very', 'closely', 'and', '[', 'event', '2', ']', 'taking', '[/', 'event', '2', ']', 'the', 'view', 'it', "'s", 'our', 'business'], ['states', 'is', 'not', 'authorized', 'to', '[', 'event', '2', ']', 'use', '[/', 'event', '2', ']', 'force', 'before', 'going', 'to', 'the', 'c'], ['are', 'as', 'expected', 'to', 'be', 'hardest', '[', 'event', '2', ']', 'hit', '[/', 'event', '2', ']', 'when', 'the', 'effects', 'of', 'the', 'As', 'i'], ['not', 'aware', 'of', 'any', 'unethical', '[', 'event', '2', ']', 'conduct', '[/', 'event', '2', '].'], ['she', 'heard', 'Amir', 'talk', 'about', '[', 'event', '2', ']', 'killing', '[/', 'event', '2', ']', 'R', 'abin', 'but', 'did', 'not', 'tell', 'th'], ['ions', 'hips', 'with', 'Russia', 'will', '[', 'event', '2', ']', 'continue', '[/', 'event', '2', ']', 'to', 'be', 'productive', 'and', 'cons'], ['r', 'stranded', 'foreigners', 'could', '[', 'event', '2', ']', 'leave', '[/', 'event', '2', ']', '``', 'as', 'soon', 'as', 'the', 'threats', 'f'], ['on', "''", 'for', 'bilateral', 'ties', 'and', '[', 'event', '2', ']', 'help', '[/', 'event', '2', ']', 'stability', 'in', 'Europe', '.', 'K', 'uch', 'ma'], ['ow', 'on', 'Thursday', 'that', 'US', 'air', '[', 'event', '2', ']', 'st', 'rikes', '[/', 'event', '2', ']', 'against', 'Iraq', 'could', 'do', 'ser'], ['l', 'ounge', 'and', 'the', 'Birmingham', '[', 'event', '2', ']', 'assault', '[/', 'event', '2', '].', '', 'An', 'intense', 'man', 'hunt', 'cond'], ['The', 'demand', 'for', 'workers', 'also', '[', 'event', '2', ']', 'led', '[/', 'event', '2', ']', 'employers', 'to', 'raise', 'wages', 'ag'], ['we', "'re", 'seeing', 'today', '.', 'That', "'s", '[', 'event', '2', ']', 'built', '[/', 'event', '2', ']', 'up', 'a', 'heck', 'of', 'a', 'momentum', 'an'], ['t', 'year', ',', 'suspects', 'had', 'been', '[', 'event', '2', ']', 'ar', 'rest', 'ed', '[/', 'event', '2', ']', 'in', 'only', 'four', 'cases', '.', 'But', 'o'], ['police', 'because', 'she', 'did', 'not', '[', 'event', '2', ']', 'bel', 'ieve', '[/', 'event', '2', ']', 'he', 'was', 'serious', '.', 'Instead', ','], ['public', '.', 'We', "'re", 'not', 'going', 'to', '[', 'event', '2', ']', 'd', 'ign', 'ify', '[/', 'event', '2', ']', 'the', 'latest', 'false', 'leak', 'wit'], ['just', 'on', 'Iraq', 'but', 'also', 'on', '[', 'event', '2', ']', 'ar', 'rest', 'ing', '[/', 'event', '2', ']', 'the', 'terrorists', 'suspected'], ['w', 'evidence', '.', '', 'Steel', 'plates', '[', 'event', '2', ']', 'rec', 'overed', '[/', 'event', '2', ']', 'at', 'the', 'Olympic', 'park', 'bomb'], ['s', 'is', 'quite', 'an', 'extraordinary', '[', 'event', '2', ']', 'story', '[/', 'event', '2', ']', 'unfolding', 'here', '.', 'We', "'ll", 'come']]
event_tokens = [['w', 'ledged', 'that', 'they', 'needed', 'to', '[', 'event', '1', ']', 'work', '[/', 'event', '1', ']', 'harder', 'to', 'solve', 'the', 'cases', 'o', '[', 'SE', 'P', ']', 'killing', 'and', 'other', 'evidence', '[', 'event', '2', ']', 'se', 'em', '[/', 'event', '2', ']', 'to', 'indicate', 'that', 'the', 'eight'], ['out', 'break', 'of', 'holy', 'war', 'could', '[', 'event', '1', ']', 'bring', '[/', 'event', '1', ']', 'thousands', 'of', 'Americans', 'hom', '[', 'SE', 'P', ']', 'de', 'firing', 'across', 'the', 'bow', 'to', '[', 'event', '2', ']', 'h', 'alt', '[/', 'event', '2', ']', 'a', 'ship', '.', 'In', 'the', 'air', ',', 'U', '.', 'S', '.', 'Ai'], ['.', '``', 'This', 'convention', 'will', 'be', '[', 'event', '1', ']', 'seen', '[/', 'event', '1', ']', 'as', 'a', 'turning', 'point', 'in', 'our', 'h', '[', 'SE', 'P', ']', 'the', 'thing', 'very', 'closely', 'and', '[', 'event', '2', ']', 'taking', '[/', 'event', '2', ']', 'the', 'view', 'it', "'s", 'our', 'business'], ['r', 'warning', 'that', 'we', 'will', 'be', '[', 'event', '1', ']', 'pre', 'pared', '[/', 'event', '1', ']', 'to', 'act', 'again', 'if', 'we', 'have', 'e', '[', 'SE', 'P', ']', 'states', 'is', 'not', 'authorized', 'to', '[', 'event', '2', ']', 'use', '[/', 'event', '2', ']', 'force', 'before', 'going', 'to', 'the', 'c'], ['t', 'amina', 'has', 'with', 'stood', 'any', '[', 'event', '1', ']', 'dis', 'rupt', 'ions', '[/', 'event', '1', ']', 'caused', 'so', 'far', 'by', 'the', 'fi', '[', 'SE', 'P', ']', 'areas', 'expected', 'to', 'be', 'hardest', '[', 'event', '2', ']', 'hit', '[/', 'event', '2', ']', 'when', 'the', 'effects', 'of', 'the', 'As', 'i'], ['est', 'ion', 'that', 'Mrs', '.', 'Cur', 'rie', 'was', '[', 'event', '1', ']', 'aware', '[/', 'event', '1', ']', 'of', 'any', 'legal', 'or', 'ethical', 'im', '[', 'SE', 'P', ']', 'not', 'aware', 'of', 'any', 'unethical', '[', 'event', '2', ']', 'conduct', '[/', 'event', '2', '].'], ['e', 'for', 'the', 'November', '1995', '[', 'event', '1', ']', 'ass', 'ass', 'ination', '[/', 'event', '1', ']', 'of', 'R', 'abin', 'at', 'a', 'Tel', 'Aviv', '[', 'SE', 'P', ']', 'she', 'heard', 'Amir', 'talk', 'about', '[', 'event', '2', ']', 'killing', '[/', 'event', '2', ']', 'R', 'abin', 'but', 'did', 'not', 'tell', 'th'], ['if', 'we', 'have', 'evidence', 'he', 'is', '[', 'event', '1', ']', 't', 'rying', '[/', 'event', '1', ']', 'to', 'rebuild', 'his', 'weapons', 'of', '[', 'SE', 'P', ']', 'ions', 'hips', 'with', 'Russia', 'will', '[', 'event', '2', ']', 'continue', '[/', 'event', '2', ']', 'to', 'be', 'productive', 'and', 'cons'], ['rd', 'an', 'arrived', 'in', 'Washington', '[', 'event', '1', ']', 'seeking', '[/', 'event', '1', ']', 'to', 'med', 'iate', 'the', 'Persian', 'Gu', '[', 'SE', 'P', ']', 'r', 'stranded', 'foreigners', 'could', '[', 'event', '2', ']', 'leave', '[/', 'event', '2', ']', '``', 'as', 'soon', 'as', 'the', 'threats', 'f'], ['how', 'the', 'disputes', 'could', 'be', '[', 'event', '1', ']', 'res', 'olved', '[/', 'event', '1', '].', 'Their', 'solution', 'would', 'req', '[', 'SE', 'P', ']', 'on', "''", 'for', 'bilateral', 'ties', 'and', '[', 'event', '2', ']', 'help', '[/', 'event', '2', ']', 'stability', 'in', 'Europe', '.', 'K', 'uch', 'ma'], ['r', 'warning', 'that', 'we', 'will', 'be', '[', 'event', '1', ']', 'pre', 'pared', '[/', 'event', '1', ']', 'to', 'act', 'again', 'if', 'we', 'have', 'e', '[', 'SE', 'P', ']', 'ow', 'on', 'Thursday', 'that', 'US', 'air', '[', 'event', '2', ']', 'st', 'rikes', '[/', 'event', '2', ']', 'against', 'Iraq', 'could', 'do', 'ser'], ['At', 'al', 'anta', 'and', 'last', 'month', "'s", '[', 'event', '1', ']', 'expl', 'osion', '[/', 'event', '1', ']', 'at', 'an', 'Alabama', 'women', "'s", 'cl', '[', 'SE', 'P', ']', 'lounge', 'and', 'the', 'Birmingham', '[', 'event', '2', ']', 'assault', '[/', 'event', '2', '].', '', 'An', 'intense', 'man', 'hunt', 'cond']]
class Adapter(nn.Module):
    # Adapter with all routing functions implemented
    def __init__(self, config, lora_config, dataset_name):
        super().__init__()
        self.input_dim = config.hidden_size
        self.down_sample_size = lora_config.r_lora
        
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 
        self.activation = ACT2FN[config.hidden_act]

        self.use_routing = lora_config.use_routing
        self.add_relu = lora_config.add_relu
        self.element_mul = lora_config.element_mul
        self.element_add = lora_config.element_add
        self.element_mul_expand = lora_config.element_mul_expand
        self.routing_xt = lora_config.routing_xt
        self.element_minus = lora_config.element_minus
    def forward(self, hidden_states, vision_hidden_states, layer_idx=None):
        residual = hidden_states
        output_states = self.down_sampler(hidden_states)
        # ipdb.set_trace()
        # for i in range(output_states.shape[0]):
        #     plot_subtoken_heatmap(vec=output_states[i].mean(-1).cpu(), routing_name='text_concatenation_adapter', index=i, title='Text Concatenation (Adapter)', subtokens=event_tokens[i], save_dir='figures/text_concatenation_adapter/')
            # plot_subtoken_heatmap(vec=output_states[i].mean(-1).cpu(), routing_name='original', index=i, title='Embedding Interaction (Adapter)', subtokens=event_tokens[i], save_dir='figures/original_false/')
        # ipdb.set_trace()
        if self.use_routing and vision_hidden_states is not None:
            vision_hidden_down = self.down_sampler(vision_hidden_states)
            if self.element_mul:       

                output_states = output_states * vision_hidden_down

                # if layer_idx == 23:
                #     for i in range(output_states.shape[0]):
                #         plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='multiplication_adapter/{}'.format(layer_idx), index=i)
                #     # plot_three_tensors_heatmap(t1=xt[i].mean(-1).cpu(), t2=xv[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='before_after_cov/after/{}'.format(layer_idx), index=i)
            # ipdb.set_trace()
                         
                # for i in range(output_states.shape[0]):
                #     plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='original_multiplication', index=i)
                # ipdb.set_trace()

            elif self.element_minus:
                # routing function: x_t - x'_v
                output_states = abs(output_states - vision_hidden_down.expand_as(output_states))
                # if layer_idx == 23:
                #     for i in range(output_states.shape[0]):
                #         plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='minus_adapter/{}'.format(layer_idx), index=i)
                
                # for i in range(output_states.shape[0]):
                #     plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='minus', index=i)
                # ipdb.set_trace()

            elif self.element_add:
                # routing function: x_t + x'_v
                output_states = output_states + vision_hidden_down.expand_as(output_states)

                # if layer_idx == 23:
                #     for i in range(output_states.shape[0]):
                #         plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='add_adapter/{}'.format(layer_idx), index=i)
                

                # for i in range(output_states.shape[0]):
                #     plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='add', index=i)
                # ipdb.set_trace()

            elif self.element_mul_expand:
                # routing function: x_t  x''_v
                # output_states = output_states @ vision_hidden_down.expand(vision_hidden_down.shape[0], self.down_sample_size, self.down_sample_size)
                output_states = output_states @ vision_hidden_down.expand_as(output_states)
            elif self.routing_xt:
                # routing function: x_t (x_t)^T  x_v, does not work very well, yet not fully explored
                output_states = output_states @ output_states.reshape(output_states.shape[0], output_states.shape[2], output_states.shape[1]) @ vision_hidden_down.expand_as(output_states)
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

# # 第一版，convolutional version
class TemporalRelationAdapter(nn.Module):
    def __init__(self, config, lora_config, dataset_name):
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

    def forward(self, event1_vec: torch.Tensor, event2_vec: torch.Tensor, layer_idx: int, context_vec: torch.Tensor = None) -> torch.Tensor:
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
        # ipdb.set_trace()
        # z = (xt.transpose(1,2) @ xv @ xt.transpose(1,2)).transpose(1,2)
        z = xt * xv
        z = self.dw_conv_3(z.transpose(1,2)).transpose(1,2)
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
        # # # ipdb.set_trace()
        # if layer_idx == 23:
        #     for i in range(e1_down.shape[0]):
        #         plot_three_tensors_heatmap(t1=e1_down[i].mean(-1).cpu(), t2=e2_down[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='conv_adapter/before/{}'.format(layer_idx), index=i)
        #         plot_three_tensors_heatmap(t1=xt[i].mean(-1).cpu(), t2=xv[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='conv_adapter/after/{}'.format(layer_idx), index=i)
        # # ipdb.set_trace()


        # 5. Apply non-linearity to the combined signal
        combined_down = self.activation(combined_down)

        # 6. Up-sample back to original hidden size
        output_delta = self.up_sampler(combined_down)
        fused_output = event1_vec + output_delta
        return fused_output

from typing import Optional, List
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_subtoken_heatmap(vec, subtokens, title=None, cmap="viridis", save_dir=None, routing_name=None, index=None):
    """
    vec: 一维张量/数组（如 torch.Tensor 或 list/np.array）
    subtokens: x 轴标签（list[str]）
    title: 可选标题
    cmap: 颜色映射（默认 'magma'）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import ipdb
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # 转为 numpy（支持 torch.Tensor 或其他可转为 array 的类型）
    try:
        v = vec.detach().cpu().numpy()
    except Exception:
        v = np.array(vec)

    v = v.reshape(1, -1)  # 单行热力图
    # ipdb.set_trace()
    subtokens = ['<s>'] + subtokens + ['<\s>']

    fig, ax = plt.subplots(figsize=(max(6, len(subtokens) * 0.3), 1.8))
    im = ax.imshow(v, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(subtokens)))
    ax.set_xticklabels(subtokens, rotation=45, ha="right")
    ax.set_yticks([])  # 不需要 y 轴
    # if title:
    # ax.set_title(title, fontsize=12, fontweight="bold")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)  # 调大 size 就更粗
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{routing_name}_{index}.png', bbox_inches='tight')
    plt.close()
    # return ax

def plot_three_tensors_heatmap(
    t1: torch.Tensor,
    t2: torch.Tensor,
    routing_name: str,
    index: int,
    subtokens1: Optional[List[str]] = None,
    subtokens2: Optional[List[str]] = None,
    title: str = "Three Tensors Heatmaps",
    cmap: str = "viridis",
    share_scale: bool = True,
    max_xticks: int = 64,
    rotate_xtick: int = 45,
    space_token: str = "▁",   # 右侧补齐用的“空格 token”
    save_dir: str = "figures",
):
    # ---------- 基本检查 ----------
    for i, t in enumerate((t1, t2), 1):
        if t.ndim != 1:
            raise ValueError(f"t{i} 必须是一维张量。")

    n1, n2 = t1.shape[0], t2.shape[0]

    def pad_or_truncate_labels(labels: Optional[List[str]], target_len: int) -> Optional[List[str]]:
        if labels is None:
            return None
        if target_len <= 0:
            return []
        L = len(labels)
        if L == target_len:
            return labels
        if L < target_len:
            return labels + [space_token] * (target_len - L)
        return labels[:target_len]

    labels1 = pad_or_truncate_labels(subtokens1, n1)
    labels2 = pad_or_truncate_labels(subtokens2, n2)

    # ---------- 转 numpy ----------
    arr1 = t1.detach().float().cpu().numpy()
    arr2 = t2.detach().float().cpu().numpy()

    # ---------- 色标范围 ----------
    if share_scale:
        vmin = float(np.nanmin([arr1.min(), arr2.min()]))
        vmax = float(np.nanmax([arr1.max(), arr2.max()]))
        # NEW: 把 space_token 位置的值设为共享的 vmin
        if labels1 is not None and len(labels1) == n1:
            mask1 = np.array(labels1, dtype=object) == space_token
            if mask1.any():
                arr1[mask1] = vmin
        if labels2 is not None and len(labels2) == n2:
            mask2 = np.array(labels2, dtype=object) == space_token
            if mask2.any():
                arr2[mask2] = vmin
    else:
        vmin = vmax = None
        # NEW: 不共享色标时，各自使用各自的最小值作为“vmin”
        if labels1 is not None and len(labels1) == n1:
            vmin1 = float(arr1.min()) if n1 > 0 else None
            if vmin1 is not None:
                mask1 = np.array(labels1, dtype=object) == space_token
                if mask1.any():
                    arr1[mask1] = vmin1
        if labels2 is not None and len(labels2) == n2:
            vmin2 = float(arr2.min()) if n2 > 0 else None
            if vmin2 is not None:
                mask2 = np.array(labels2, dtype=object) == space_token
                if mask2.any():
                    arr2[mask2] = vmin2

    # ---------- 画布 ----------
    width_units = max(n1, n2)
    fig_w = float(np.clip(width_units / 12.0, 8.0, 20.0))
    fig_h = 2.2 + 0.8
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), constrained_layout=True)
    ax1, ax2 = axes

    def _set_xticks_with_labels(ax, N, labels: Optional[List[str]]):
        if N == 0:
            ax.set_xticks([])
            return
        if N <= max_xticks:
            xticks = np.arange(N)
        else:
            step = max(1, N // max_xticks)
            xticks = np.arange(0, N, step)
        ax.set_xticks(xticks)
        if labels is not None:
            ax.set_xticklabels([labels[i] for i in xticks], rotation=rotate_xtick, ha="right")
        else:
            ax.set_xticklabels([str(i) for i in xticks])

    # ---------- 第一张 ----------
    im1 = ax1.imshow(arr1[np.newaxis, :], aspect="auto", interpolation="nearest",
                     cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_yticks([0])
    ax1.set_yticklabels(["event1"])
    _set_xticks_with_labels(ax1, n1, labels1)
    ax1.set_xlabel("subtokens" if labels1 is not None else "index")
    # ax1.set_title(f"{title} — 1")

    # ---------- 第二张 ----------
    im2 = ax2.imshow(arr2[np.newaxis, :], aspect="auto", interpolation="nearest",
                     cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["event2"])
    _set_xticks_with_labels(ax2, n2, labels2)
    ax2.set_xlabel("subtokens" if labels2 is not None else "index")
    # ax2.set_title(f"{title} — 2")

    if share_scale:
        cbar = fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("value")
    else:
        for im, ax in zip((im1, im2), axes):
            cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            cbar.set_label("value")

    # ---------- 保存 ----------
    out_dir = os.path.join(save_dir, routing_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heatmap_{index}.png")
    # fig.suptitle(title, y=1.02, fontsize=12)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path



# def plot_three_tensors_heatmap(
#     t1: torch.Tensor,
#     t2: torch.Tensor,
#     routing_name: str,
#     index: int,
#     subtokens1: Optional[List[str]] = None,
#     subtokens2: Optional[List[str]] = None,
#     title: str = "Three Tensors Heatmaps",
#     cmap: str = "viridis",
#     share_scale: bool = True,
#     max_xticks: int = 64,
#     rotate_xtick: int = 45,
#     space_token: str = "▁",   # 右侧补齐用的“空格 token”
#     save_dir: str = "figures",
# ):
#     """
#     三个一维张量的热力图（竖排）：
#       - 第一张：t1，x 轴为 subtokens1（若长度不匹配则左对齐，右侧用 space_token 补；更长则截断；为空则用 index）
#       - 第二张：t2，同上（subtokens2）
#       - 第三张：t3，x 轴为 index（0..N-1）
#     三者长度可不相同。
#     """

#     # ---------- 基本检查 ----------
#     for i, t in enumerate((t1, t2), 1):
#         if t.ndim != 1:
#             raise ValueError(f"t{i} 必须是一维张量。")

#     n1, n2 = t1.shape[0], t2.shape[0]

#     def pad_or_truncate_labels(labels: Optional[List[str]], target_len: int) -> Optional[List[str]]:
#         """左对齐：短则右侧用 space_token 补齐，长则截断；labels 为 None 则返回 None。"""
#         if labels is None:
#             return None
#         if target_len <= 0:
#             return []
#         L = len(labels)
#         if L == target_len:
#             return labels
#         if L < target_len:
#             return labels + [space_token] * (target_len - L)
#         # L > target_len -> 截断（左对齐）
#         return labels[:target_len]

#     labels1 = pad_or_truncate_labels(subtokens1, n1)
#     labels2 = pad_or_truncate_labels(subtokens2, n2)

#     # ---------- 转 numpy ----------
#     arr1 = t1.detach().float().cpu().numpy()
#     arr2 = t2.detach().float().cpu().numpy()

#     # ---------- 色标范围 ----------
#     if share_scale:
#         vmin = float(np.nanmin([arr1.min(), arr2.min()]))
#         vmax = float(np.nanmax([arr1.max(), arr2.max()]))
#     else:
#         vmin = vmax = None

#     # ---------- 画布 ----------
#     width_units = max(n1, n2)
#     fig_w = float(np.clip(width_units / 12.0, 8.0, 20.0))
#     fig_h = 2.2 + 2.2 + 2.4
#     fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), constrained_layout=True)
#     ax1, ax2 = axes

#     def _set_xticks_with_labels(ax, N, labels: Optional[List[str]]):
#         """抽样刻度；labels=None 时用数字 index。"""
#         if N == 0:
#             ax.set_xticks([])
#             return
#         if N <= max_xticks:
#             xticks = np.arange(N)
#         else:
#             step = max(1, N // max_xticks)
#             xticks = np.arange(0, N, step)
#         ax.set_xticks(xticks)
#         if labels is not None:
#             ax.set_xticklabels([labels[i] for i in xticks], rotation=rotate_xtick, ha="right")
#         else:
#             ax.set_xticklabels([str(i) for i in xticks])

#     # ---------- 第一张 ----------
#     im1 = ax1.imshow(arr1[np.newaxis, :], aspect="auto", interpolation="nearest",
#                      cmap=cmap, vmin=vmin, vmax=vmax)
#     ax1.set_yticks([0])
#     ax1.set_yticklabels(["event1"])
#     _set_xticks_with_labels(ax1, n1, labels1)
#     ax1.set_xlabel("subtokens" if labels1 is not None else "index")
#     ax1.set_title(f"{title} — 1")

#     # ---------- 第二张 ----------
#     im2 = ax2.imshow(arr2[np.newaxis, :], aspect="auto", interpolation="nearest",
#                      cmap=cmap, vmin=vmin, vmax=vmax)
#     ax2.set_yticks([0])
#     ax2.set_yticklabels(["event2"])
#     _set_xticks_with_labels(ax2, n2, labels2)
#     ax2.set_xlabel("subtokens" if labels2 is not None else "index")
#     ax2.set_title(f"{title} — 2")

#     if share_scale:
#         cbar = fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.02)
#         cbar.set_label("value")
#     else:
#         for im, ax in zip((im1, im2), axes):
#             cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
#             cbar.set_label("value")

#     # ---------- 保存 ----------
#     out_dir = os.path.join(save_dir, routing_name)
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f"heatmap_{index}.png")
#     fig.suptitle(title, y=1.02, fontsize=12)
#     plt.savefig(out_path, bbox_inches="tight", dpi=200)
#     plt.close(fig)
#     return out_path



# def plot_one_tensors_heatmap(t1: torch.Tensor,
#                                index: int,
#                                title: str = "Three Tensors Heatmap",
#                                cmap: str = "viridis",
#                                share_scale: bool = True):
#     N = t1.shape[0]

#     arrs = [t.detach().float().cpu().numpy() for t in (t1)]
#     data = np.stack(arrs, axis=0)  # (3, N)

#     vmin = data.min() if share_scale else None
#     vmax = data.max() if share_scale else None

#     plt.figure(figsize=(12, 1.0))
#     im = plt.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap,
#                     vmin=vmin, vmax=vmax)
#     plt.colorbar(im, fraction=0.02, pad=0.02, label="value")

#     plt.yticks([0], ["tensor 1"])

#     # 根据长度自适应 x 轴刻度（512 时每 64 一个刻度）
#     step = max(1, N // 8)
#     ticks = list(range(0, N, step))
#     plt.xticks(ticks, [str(i) for i in ticks])

#     plt.xlabel(f"index (0 … {N-1})")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(f'figures/three_tensors_heatmap_{index}.png')


# import math
# import torch
# import torch.nn as nn
# import ipdb
# # MMA-MoReS
# class TemporalRelationAdapter(nn.Module):
#     def __init__(self, config, lora_config, dataset_name):
#         super().__init__()
#         self.input_dim = config.hidden_size
#         self.down_size = lora_config.r_lora  # bottleneck rank d

#         # 低秩线性投影（无 bias，纯线性）
#         self.down_sampler = nn.Linear(self.input_dim, self.down_size, bias=False)  # W_down
#         self.up_sampler   = nn.Linear(self.down_size, self.input_dim, bias=False)  # W_up

#         # 论文里的 Linear(h) 项：R^D -> R^d（无激活）
#         self.linear_h = nn.Linear(self.input_dim, self.down_size, bias=False)

#     def forward(self, event1_vec: torch.Tensor,
#                       event2_vec: torch.Tensor,
#                       context_vec: torch.Tensor = None) -> torch.Tensor:

#         # 1) 降维（W_down h）
#         e1_down = self.down_sampler(event2_vec)  # [B, d]
#         # ipdb.set_trace()

#         # 2) 子空间 steering：φ(h) = Linear(h) - W_down h
#         phi = self.linear_h(event2_vec) - e1_down  # [B, d]

#         # 3) 升维并 residual 相加：h' = h + W_up φ(h)
#         delta = self.up_sampler(phi)               # [B, D]
#         fused_output = event1_vec + delta          # [B, D]

#         return fused_output


# import math
# import torch
# import torch.nn as nn

# class TemporalRelationAdapter(nn.Module):
#     def __init__(self, config, lora_config, dataset_name):
#         super().__init__()
#         self.input_dim = config.hidden_size
#         self.down_size = lora_config.r_lora  # bottleneck dim

#         # 保留你原结构
#         self.down_sampler = nn.Linear(self.input_dim, self.down_size)
#         self.up_sampler   = nn.Linear(self.down_size, self.input_dim)
#         self.activation   = ACT2FN[config.hidden_act]

#         # 低秩 CP 因子（两路输入各一套），形状 [R, d, d]
#         self.rank_h = getattr(lora_config, "r_cp",
#                               max(4, self.down_size // 8))
#         def make_cp_factors():
#             w = nn.Parameter(torch.empty(self.rank_h, self.down_size, self.down_size))
#             nn.init.xavier_uniform_(w)
#             return w
#         self.W_e1 = make_cp_factors()
#         self.W_e2 = make_cp_factors()

#     @staticmethod
#     def _apply_cp_bt(z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
#         """
#         z: [B, T, d]; W: [R, d, d]
#         返回: sum_r (W_r @ z) 逐 token 线性映射，形状仍为 [B, T, d]
#         """
#         # einsum: (r k d, b t d) -> (b r t k) ，再对 r 求和 -> (b t k)
#         tmp = torch.einsum('rkd,btd->brtk', W, z)  # [B, R, T, d]
#         return tmp.sum(dim=1)  # [B, T, d]

#     def forward(self, event1_vec: torch.Tensor, event2_vec: torch.Tensor) -> torch.Tensor:
#         # 输入: [B, T, D]，两者同形状（按你的要求不再检查）
#         # 1) Down 到 bottleneck 子空间（逐 token 线性）
#         e1_down = self.down_sampler(event1_vec)  # [B, T, d]
#         e2_down = self.down_sampler(event2_vec)  # [B, T, d]

#         # 2) 低秩因子 + 逐元素乘的显式 routing（无 NN 路由网络）
#         z1 = self._apply_cp_bt(e1_down, self.W_e1)  # [B, T, d]
#         z2 = self._apply_cp_bt(e2_down, self.W_e2)  # [B, T, d]
#         combined_down = z1 * z2                    # element-wise product

#         # 3) 激活（保持你原流程：融合后激活）
#         combined_down = self.activation(combined_down)

#         # 4) Up 回 hidden，并 residual 到 event1
#         output_delta = self.up_sampler(combined_down)    # [B, T, D]
#         fused_output = event1_vec + output_delta         # [B, T, D]
#         return fused_output






# class TemporalRelationAdapter(nn.Module):
#     def __init__(self, config, lora_config, dataset_name):
#         super().__init__()
#         self.input_dim = config.hidden_size
#         self.down_size = lora_config.r_lora  # bottleneck dimension (low-rank size)

#         # Low-rank projections
#         self.down_sampler = nn.Linear(self.input_dim, self.down_size)
#         self.up_sampler   = nn.Linear(self.down_size, self.input_dim)
#         # Non-linear activation function
#         self.activation = ACT2FN[config.hidden_act]

#     def forward(self, event1_vec: torch.Tensor, event2_vec: torch.Tensor, context_vec: torch.Tensor = None) -> torch.Tensor:
#         # 1. Down-sample both events (and context if provided) to bottleneck dimension
#         e1_down = self.down_sampler(event1_vec)  # [batch, down_size]
#         e2_down = self.down_sampler(event2_vec)  # [batch, down_size]
#         combined_down = e1_down * e2_down
#         combined_down = self.activation(combined_down)

#         # 6. Up-sample back to original hidden size
#         output_delta = self.up_sampler(combined_down)
#         fused_output = event1_vec + output_delta
#         return fused_output

# 第二版，local self-attention version
# class TemporalRelationAdapter(nn.Module):
#     def __init__(self, config, lora_config, dataset_name,
#                  window_size=5, num_heads=1, attn_dropout=0.0):
#         super().__init__()
#         self.input_dim = config.hidden_size
#         self.down_size = lora_config.r_lora  # bottleneck dimension (low-rank size)

#         # Low-rank projections
#         self.down_sampler = nn.Linear(self.input_dim, self.down_size)
#         self.up_sampler   = nn.Linear(self.down_size, self.input_dim)

#         # Non-linear activation function
#         self.activation = ACT2FN[config.hidden_act]

#         # === 用局部 self-attention 替换 depthwise conv（无 LayerNorm）
#         self.window_size = window_size
#         self.attn1 = nn.MultiheadAttention(self.down_size, num_heads, dropout=attn_dropout, batch_first=True)
#         self.attn2 = nn.MultiheadAttention(self.down_size, num_heads, dropout=attn_dropout, batch_first=True)
#         self.attn3 = nn.MultiheadAttention(self.down_size, num_heads, dropout=attn_dropout, batch_first=True)

#     def _build_local_attn_mask(self, L, device, dtype):
#         """
#         仅允许 |i-j| <= w 的注意力；其它位置设为 -inf。
#         形状 [L, L]，作为 additive mask 传入 nn.MultiheadAttention。
#         """
#         w = self.window_size
#         idx = torch.arange(L, device=device)
#         dist = idx[None, :] - idx[:, None]  # [L, L]
#         far = (dist.abs() > w)
#         mask = torch.zeros((L, L), device=device, dtype=dtype)
#         mask.masked_fill_(far, float("-inf"))
#         return mask

#     def forward(self, event1_vec: torch.Tensor, event2_vec: torch.Tensor, context_vec: torch.Tensor = None) -> torch.Tensor:
#         # 1) 下采样
#         e1_down = self.down_sampler(event1_vec)  # [B, L1, d]
#         e2_down = self.down_sampler(event2_vec)  # [B, L2, d]

#         B, L1, d = e1_down.shape
#         B2, L2, d2 = e2_down.shape
#         assert d == d2 and B == B2, "e1/e2 维度或 batch 不匹配"

#         # 2) 窗口 self-attention（替换 dw_conv_1 / dw_conv_2），保留残差
#         mask1 = self._build_local_attn_mask(L1, e1_down.device, e1_down.dtype)
#         xt_attn, _ = self.attn1(e1_down, e1_down, e1_down, attn_mask=mask1, need_weights=False)
#         # xt = e1_down + xt_attn
#         xt = xt_attn

#         mask2 = self._build_local_attn_mask(L2, e2_down.device, e2_down.dtype)
#         xv_attn, _ = self.attn2(e2_down, e2_down, e2_down, attn_mask=mask2, need_weights=False)
#         xv = xv_attn

#         # 3) 逐点乘并对齐长度（与原逻辑一致）
#         L = min(L1, L2)
#         xt = xt[:, :L, :]
#         xv = xv[:, :L, :]
#         z = xt * xv  # [B, L, d]

#         # 4) 第三层窗口 self-attention（替换 dw_conv_3），保留残差
#         mask3 = self._build_local_attn_mask(L, z.device, z.dtype)
#         z_attn, _ = self.attn3(z, z, z, attn_mask=mask3, need_weights=False)
#         # combined_down = z + z_attn
#         combined_down = z_attn

#         # 5) 激活 + 上采样 + 残差回加
#         combined_down = self.activation(combined_down)
#         output_delta = self.up_sampler(combined_down)      # [B, L, H]
#         fused_output = event1_vec[:, :L, :] + output_delta
#         return fused_output

# class TemporalRelationAdapter(nn.Module):
#     """
#     Rotary-enhanced Temporal Relation Adapter.
#     假设输入形状: [batch, seq_len, hidden_size]
#     - 将 event1_vec / event2_vec 下采样到 bottleneck 维度
#     - 在逐元素乘法前，对两者分别施加 RoPE（按序号 i 旋转）
#     - 之后的流程与你原来一致：depthwise conv -> 非线性 -> 上采样 -> 残差到 event1
#     """
#     def __init__(self, config, lora_config, dataset_name=None, rope_base: float = 10000.0):
#         super().__init__()
#         self.input_dim = config.hidden_size
#         self.down_size = int(lora_config.r_lora)  # bottleneck (需要为偶数以配对旋转)
#         assert self.down_size % 2 == 0, "RoPE 需要偶数维度（down_size 必须为偶数）"

#         # === Low-rank projections ===
#         self.down_sampler = nn.Linear(self.input_dim, self.down_size)
#         self.up_sampler   = nn.Linear(self.down_size, self.input_dim)

#         # === Non-linearity ===
#         self.activation = ACT2FN[config.hidden_act]

#         # === Depthwise 1D convs (channels-last -> channels-first -> channels-last) ===
#         # groups=channels 实现 depthwise
#         self.dw_conv_1 = nn.Conv1d(self.down_size, self.down_size, kernel_size=5, padding=2, groups=self.down_size)
#         self.dw_conv_2 = nn.Conv1d(self.down_size, self.down_size, kernel_size=5, padding=2, groups=self.down_size)
#         self.dw_conv_3 = nn.Conv1d(self.down_size, self.down_size, kernel_size=5, padding=2, groups=self.down_size)

#         # === RoPE parameters & cache ===
#         self.rope_base = float(rope_base)
#         # inv_freq 形状: [down_size/2]
#         dim_half = self.down_size // 2
#         # θ_i = base^{-2i/d}; 这里用 inv_freq = base^{-2i/d} 的对数形式避免数值不稳定
#         # 常见实现是 inv_freq = 1 / (base ** (2i/d))
#         inv_exponent = torch.arange(0, dim_half, dtype=torch.float32) * (2.0 / self.down_size)
#         inv_freq = self.rope_base ** (-inv_exponent)  # [dim_half]
#         self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
#         # 运行期按 seq_len 动态缓存 cos/sin
#         self.register_buffer("_rope_cos_cached", torch.empty(0), persistent=False)
#         self.register_buffer("_rope_sin_cached", torch.empty(0), persistent=False)
#         self._rope_seq_len_cached = 0

#     @torch.no_grad()
#     def _build_rope_cache(self, seq_len: int, device, dtype):
#         """
#         按当前 seq_len 构建/更新 cos/sin 缓存:
#         - cos/sin 形状: [seq_len, 1, down_size] 以便对 [B, S, D] 广播
#         """
#         if self._rope_seq_len_cached >= seq_len \
#            and self._rope_cos_cached.device == device \
#            and self._rope_cos_cached.dtype == dtype:
#             return  # 已经足够长 & 设备/类型匹配

#         pos = torch.arange(seq_len, device=device, dtype=self.rope_inv_freq.dtype)  # [S]
#         # freqs: [S, D/2]
#         freqs = torch.einsum('s,d->sd', pos, self.rope_inv_freq)  # 外积：s*d/2
#         # 展开到 [S, D]：偶/奇交替
#         # cos/sin 对每个二维子空间（偶/奇维）复用
#         cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [S, D]
#         sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [S, D]

#         # reshape 成便于广播的形状 [S, 1, D]
#         self._rope_cos_cached = cos.to(dtype=dtype, device=device).unsqueeze(1)  # [S,1,D]
#         self._rope_sin_cached = sin.to(dtype=dtype, device=device).unsqueeze(1)  # [S,1,D]
#         self._rope_seq_len_cached = seq_len

#     def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         对 x 应用 RoPE 旋转。
#         x: [B, S, D]  (D = down_size, 偶数)
#         返回: 同形状 [B, S, D]
#         公式（逐二维子空间）:
#             x_even' =  x_even * cos - x_odd * sin
#             x_odd'  =  x_even * sin + x_odd * cos
#         """
#         B, S, D = x.shape
#         self._build_rope_cache(S, x.device, x.dtype)
#         cos = self._rope_cos_cached[:S]  # [S,1,D]
#         sin = self._rope_sin_cached[:S]  # [S,1,D]

#         # 广播到 [B,S,D]
#         # 偶/奇切分
#         x_even = x[..., 0::2]   # [B,S,D/2]
#         x_odd  = x[..., 1::2]   # [B,S,D/2]
#         cos_e  = cos[..., 0::2].expand(B, S, -1)  # [B,S,D/2]
#         sin_e  = sin[..., 0::2].expand(B, S, -1)  # [B,S,D/2]

#         # 旋转
#         x_even_rot = x_even * cos_e - x_odd * sin_e
#         x_odd_rot  = x_even * sin_e + x_odd * cos_e

#         # 交错还原到 [B,S,D]
#         x_rot = torch.empty_like(x)
#         x_rot[..., 0::2] = x_even_rot
#         x_rot[..., 1::2] = x_odd_rot
#         return x_rot

#     def forward(
#         self,
#         event1_vec: torch.Tensor,   # [B, S, H]
#         event2_vec: torch.Tensor,   # [B, S, H] (或 [B, S2, H]，若不同 S 会在 conv 前报错)
#         context_vec: torch.Tensor = None  # 可选 [B, S, H]，当前实现不强依赖
#     ) -> torch.Tensor:
#         # 1) bottleneck 下采样
#         #    线性层逐 token 处理 -> [B, S, D]
#         e1_down = self.down_sampler(event1_vec)
#         e2_down = self.down_sampler(event2_vec)

#         # 2) 在交互前施加 RoPE（关键！）
#         #    把 e1 当作 q，把 e2 当作 k —— attention 中应在内积前编码
#         e1_down = self._apply_rope(e1_down)  # [B,S,D]
#         e2_down = self._apply_rope(e2_down)  # [B,S,D]

#         # 3) depthwise conv（保持与你原实现对齐）
#         #    Conv1d 期望 [B, C, S]，我们把 D 当作通道
#         xt = self.dw_conv_1(e1_down.transpose(1, 2)).transpose(1, 2)  # [B,S,D]
#         xv = self.dw_conv_2(e2_down.transpose(1, 2)).transpose(1, 2)  # [B,S,D]

#         # 4) 逐元素乘作为交互（你的设计）
#         #    注意：RoPE 的相对位置信息已“注入”，乘积 & 后续卷积会对这种位移敏感
#         z = xt * xv  # [B,S,D]
#         z = self.dw_conv_3(z.transpose(1, 2)).transpose(1, 2)  # [B,S,D]

#         # 5) 非线性
#         z = self.activation(z)

#         # 6) 上采样 + 残差
#         output_delta = self.up_sampler(z)  # [B,S,H]
#         fused_output = event1_vec + output_delta
#         return fused_output