import math
from typing import Optional, Union, List

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import loralib.layers_encoder as loralayer

from transformers.activations import ACT2FN, gelu
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import RobertaConfig
# import matplotlib.pyplot as plt
import numpy as np
import ipdb
import os

logger = logging.get_logger(__name__)

# Add missing functions for direct import
def auto_docstring(func):
    """Decorator for auto-docstring (placeholder)"""
    return func

def get_torch_version():
    """Get torch version (placeholder)"""
    import torch
    return torch.__version__


# ============================
# event_1_tokens = [['w', 'ledged', 'that', 'they', 'needed', 'to', '[', 'event', '1', ']', 'work', '[/', 'event', '1', ']', 'harder', 'to', 'solve', 'the', 'cases', 'o'], ['out', 'break', 'of', 'holy', 'war', 'could', '[', 'event', '1', ']', 'bring', '[/', 'event', '1', ']', 'thousands', 'of', 'Americans', 'hom'], ['.', '``', 'This', 'convention', 'will', 'be', '[', 'event', '1', ']', 'seen', '[/', 'event', '1', ']', 'as', 'a', 'turning', 'point', 'in', 'our', 'h'], ['r', 'warning', 'that', 'we', 'will', 'be', '[', 'event', '1', ']', 'pre', 'pared', '[/', 'event', '1', ']', 'to', 'act', 'again', 'if', 'we', 'have', 'e'], ['t', 'amina', 'has', 'with', 'stood', 'any', '[', 'event', '1', ']', 'dis', 'rupt', 'ions', '[/', 'event', '1', ']', 'caused', 'so', 'far', 'by', 'the', 'fi'], ['est', 'ion', 'that', 'Mrs', '.', 'Cur', 'rie', 'was', '[', 'event', '1', ']', 'aware', '[/', 'event', '1', ']', 'of', 'any', 'legal', 'or', 'ethical', 'im'], ['e', 'for', 'the', 'November', '1995', '[', 'event', '1', ']', 'ass', 'ass', 'ination', '[/', 'event', '1', ']', 'of', 'R', 'abin', 'at', 'a', 'Tel', 'Aviv'], ['if', 'we', 'have', 'evidence', 'he', 'is', '[', 'event', '1', ']', 't', 'rying', '[/', 'event', '1', ']', 'to', 'rebuild', 'his', 'weapons', 'of'], ['rd', 'an', 'arrived', 'in', 'Washington', '[', 'event', '1', ']', 'seeking', '[/', 'event', '1', ']', 'to', 'med', 'iate', 'the', 'Persian', 'Gu'], ['how', 'the', 'disputes', 'could', 'be', '[', 'event', '1', ']', 'res', 'olved', '[/', 'event', '1', '].', 'Their', 'solution', 'would', 'req'], ['r', 'warning', 'that', 'we', 'will', 'be', '[', 'event', '1', ']', 'pre', 'pared', '[/', 'event', '1', ']', 'to', 'act', 'again', 'if', 'we', 'have', 'e'], ['At', 'al', 'anta', 'and', 'last', 'month', "'s", '[', 'event', '1', ']', 'expl', 'osion', '[/', 'event', '1', ']', 'at', 'an', 'Alabama', 'women', "'s", 'cl'], ['t', 'amina', 'has', 'with', 'stood', 'any', '[', 'event', '1', ']', 'dis', 'rupt', 'ions', '[/', 'event', '1', ']', 'caused', 'so', 'far', 'by', 'the', 'fi'], ['Iraq', ',', 'oil', 'prices', 'have', 'been', '[', 'event', '1', ']', 'fall', 'ing', '[/', 'event', '1', '],', 'that', "'s", 'because', 'of', 'a', 'wor', 'l'], ['ro', 'secut', 'ors', 'said', 'they', 'had', '[', 'event', '1', ']', 'identified', '[/', 'event', '1', ']', 'different', 'suspects', 'in', 'si'], ['f', 'R', 'abin', 'at', 'a', 'Tel', 'Aviv', 'peace', '[', 'event', '1', ']', 'r', 'ally', '[/', 'event', '1', '].', 'Newspaper', 'reports', 'have', 'sa'], ['est', 'ion', 'that', 'Mrs', '.', 'Cur', 'rie', 'was', '[', 'event', '1', ']', 'aware', '[/', 'event', '1', ']', 'of', 'any', 'legal', 'or', 'ethical', 'im'], ['Clinton', 'discussed', 'the', 'Iraq', '[', 'event', '1', ']', 'c', 'risis', '[/', 'event', '1', ']', 'with', 'the', 'one', 'ally', 'who', 'has'], ['ma', 'women', "'s", 'clinic', 'might', 'be', '[', 'event', '1', ']', 'related', '[/', 'event', '1', '].', 'Pierre', 'Thomas', 'has', 'the', 'la'], ['ast', '.', '', 'But', 'now', 'we', "'re", 'going', 'to', '[', 'event', '1', ']', 'go', '[/', 'event', '1', ']', 'back', 'to', 'New', 'York', ',', 'where', 'Dian']]
# event_2_tokens = [['killing', 'and', 'other', 'evidence', '[', 'event', '2', ']', 'se', 'em', '[/', 'event', '2', ']', 'to', 'indicate', 'that', 'the', 'eight'], ['de', 'firing', 'across', 'the', 'bow', 'to', '[', 'event', '2', ']', 'h', 'alt', '[/', 'event', '2', ']', 'a', 'ship', '.', 'In', 'the', 'air', ',', 'U', '.', 'S', '.', 'Ai'], ['the', 'thing', 'very', 'closely', 'and', '[', 'event', '2', ']', 'taking', '[/', 'event', '2', ']', 'the', 'view', 'it', "'s", 'our', 'business'], ['states', 'is', 'not', 'authorized', 'to', '[', 'event', '2', ']', 'use', '[/', 'event', '2', ']', 'force', 'before', 'going', 'to', 'the', 'c'], ['are', 'as', 'expected', 'to', 'be', 'hardest', '[', 'event', '2', ']', 'hit', '[/', 'event', '2', ']', 'when', 'the', 'effects', 'of', 'the', 'As', 'i'], ['not', 'aware', 'of', 'any', 'unethical', '[', 'event', '2', ']', 'conduct', '[/', 'event', '2', '].'], ['she', 'heard', 'Amir', 'talk', 'about', '[', 'event', '2', ']', 'killing', '[/', 'event', '2', ']', 'R', 'abin', 'but', 'did', 'not', 'tell', 'th'], ['ions', 'hips', 'with', 'Russia', 'will', '[', 'event', '2', ']', 'continue', '[/', 'event', '2', ']', 'to', 'be', 'productive', 'and', 'cons'], ['r', 'stranded', 'foreigners', 'could', '[', 'event', '2', ']', 'leave', '[/', 'event', '2', ']', '``', 'as', 'soon', 'as', 'the', 'threats', 'f'], ['on', "''", 'for', 'bilateral', 'ties', 'and', '[', 'event', '2', ']', 'help', '[/', 'event', '2', ']', 'stability', 'in', 'Europe', '.', 'K', 'uch', 'ma'], ['ow', 'on', 'Thursday', 'that', 'US', 'air', '[', 'event', '2', ']', 'st', 'rikes', '[/', 'event', '2', ']', 'against', 'Iraq', 'could', 'do', 'ser'], ['l', 'ounge', 'and', 'the', 'Birmingham', '[', 'event', '2', ']', 'assault', '[/', 'event', '2', '].', '', 'An', 'intense', 'man', 'hunt', 'cond'], ['The', 'demand', 'for', 'workers', 'also', '[', 'event', '2', ']', 'led', '[/', 'event', '2', ']', 'employers', 'to', 'raise', 'wages', 'ag'], ['we', "'re", 'seeing', 'today', '.', 'That', "'s", '[', 'event', '2', ']', 'built', '[/', 'event', '2', ']', 'up', 'a', 'heck', 'of', 'a', 'momentum', 'an'], ['t', 'year', ',', 'suspects', 'had', 'been', '[', 'event', '2', ']', 'ar', 'rest', 'ed', '[/', 'event', '2', ']', 'in', 'only', 'four', 'cases', '.', 'But', 'o'], ['police', 'because', 'she', 'did', 'not', '[', 'event', '2', ']', 'bel', 'ieve', '[/', 'event', '2', ']', 'he', 'was', 'serious', '.', 'Instead', ','], ['public', '.', 'We', "'re", 'not', 'going', 'to', '[', 'event', '2', ']', 'd', 'ign', 'ify', '[/', 'event', '2', ']', 'the', 'latest', 'false', 'leak', 'wit'], ['just', 'on', 'Iraq', 'but', 'also', 'on', '[', 'event', '2', ']', 'ar', 'rest', 'ing', '[/', 'event', '2', ']', 'the', 'terrorists', 'suspected'], ['w', 'evidence', '.', '', 'Steel', 'plates', '[', 'event', '2', ']', 'rec', 'overed', '[/', 'event', '2', ']', 'at', 'the', 'Olympic', 'park', 'bomb'], ['s', 'is', 'quite', 'an', 'extraordinary', '[', 'event', '2', ']', 'story', '[/', 'event', '2', ']', 'unfolding', 'here', '.', 'We', "'ll", 'come']]

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
    def forward(self, hidden_states, vision_hidden_states):
        residual = hidden_states
        output_states = self.down_sampler(hidden_states)

        # for i in range(output_states.shape[0]):
        #     plot_three_tensors_heatmap(output_states[i].mean(-1).cpu(), vision_hidden_down[i].mean(-1).cpu(), (output_states[i] * vision_hidden_down[i]).mean(-1).cpu(), i)
        # ipdb.set_trace()
        if self.use_routing and vision_hidden_states is not None:
            vision_hidden_down = self.down_sampler(vision_hidden_states)
            if self.element_mul:       

                output_states = output_states * vision_hidden_down
                         
                # for i in range(output_states.shape[0]):
                #     plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), t3=(output_states[i] * vision_hidden_down[i]).mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='original_multiplication', index=i)
                # ipdb.set_trace()

            elif self.element_minus:
                # routing function: x_t - x'_v
                output_states = output_states - vision_hidden_down.expand_as(output_states)

                # for i in range(output_states.shape[0]):
                #     plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), t3=(output_states[i] - vision_hidden_down[i]).mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='minus', index=i)
                # ipdb.set_trace()

            elif self.element_add:
                # routing function: x_t + x'_v
                output_states = output_states + vision_hidden_down.expand_as(output_states)

                for i in range(output_states.shape[0]):
                    plot_three_tensors_heatmap(t1=output_states[i].mean(-1).cpu(), t2=vision_hidden_down[i].mean(-1).cpu(), t3=(output_states[i] + vision_hidden_down[i]).mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='add', index=i)
                ipdb.set_trace()

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


# def plot_three_tensors_heatmap(t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, routing_name: str,
#                                index: int,
#                                title: str = "Three Tensors Heatmap",
#                                cmap: str = "viridis",
#                                share_scale: bool = True):
#     """
#     将三个一维 tensor 画在同一张热力图上（3 行 × N 列），N 可为 512 或任意长度。
#     """
#     # 基本检查：都必须是一维，且长度一致
#     if not all(t.ndim == 1 for t in (t1, t2, t3)):
#         raise ValueError("所有输入必须为一维张量。")
#     if not (t1.shape[0] == t2.shape[0] == t3.shape[0]):
#         raise ValueError("三个张量长度必须一致。")
#     N = t1.shape[0]

#     arrs = [t.detach().float().cpu().numpy() for t in (t1, t2, t3)]
#     data = np.stack(arrs, axis=0)  # (3, N)

#     vmin = data.min() if share_scale else None
#     vmax = data.max() if share_scale else None

#     plt.figure(figsize=(12, 3.0))
#     im = plt.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap,
#                     vmin=vmin, vmax=vmax)
#     plt.colorbar(im, fraction=0.02, pad=0.02, label="value")

#     plt.yticks([0, 1, 2], ["event1", "event2", "interaction"])

#     # 根据长度自适应 x 轴刻度（512 时每 64 一个刻度）
#     step = max(1, N // 8)
#     ticks = list(range(0, N, step))
#     plt.xticks(ticks, [str(i) for i in ticks])

#     plt.xlabel(f"index (0 … {N-1})")
#     plt.title(title)
#     plt.tight_layout()
#     if os.path.exists(f'figures/{routing_name}') is False:
#         os.makedirs(f'figures/{routing_name}')
#     plt.savefig(f'figures/{routing_name}/heatmap_{index}.png')



# def plot_three_tensors_heatmap(
#     t1: torch.Tensor,
#     t2: torch.Tensor,
#     t3: torch.Tensor,
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
#     for i, t in enumerate((t1, t2, t3), 1):
#         if t.ndim != 1:
#             raise ValueError(f"t{i} 必须是一维张量。")

#     n1, n2, n3 = t1.shape[0], t2.shape[0], t3.shape[0]

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
#     arr3 = t3.detach().float().cpu().numpy()

#     # ---------- 色标范围 ----------
#     if share_scale:
#         vmin = float(np.nanmin([arr1.min(), arr2.min(), arr3.min()]))
#         vmax = float(np.nanmax([arr1.max(), arr2.max(), arr3.max()]))
#     else:
#         vmin = vmax = None

#     # ---------- 画布 ----------
#     width_units = max(n1, n2, n3)
#     fig_w = float(np.clip(width_units / 12.0, 8.0, 20.0))
#     fig_h = 2.2 + 2.2 + 2.4
#     fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), constrained_layout=True)
#     ax1, ax2, ax3 = axes

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

#     # ---------- 第三张（index） ----------
#     im3 = ax3.imshow(arr3[np.newaxis, :], aspect="auto", interpolation="nearest",
#                      cmap=cmap, vmin=vmin, vmax=vmax)
#     ax3.set_yticks([0])
#     ax3.set_yticklabels(["interaction"])
#     _set_xticks_with_labels(ax3, n3, labels=None)
#     ax3.set_xlabel("index (0 … %d)" % (n3 - 1 if n3 > 0 else 0))
#     ax3.set_title(f"{title} — 3")

#     # ---------- 色条 ----------
#     if share_scale:
#         cbar = fig.colorbar(im3, ax=axes, fraction=0.02, pad=0.02)
#         cbar.set_label("value")
#     else:
#         for im, ax in zip((im1, im2, im3), axes):
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
#     """
#     将三个一维 tensor 画在同一张热力图上（3 行 × N 列），N 可为 512 或任意长度。
#     """
#     # 基本检查：都必须是一维，且长度一致
#     # if not all(t.ndim == 1 for t in (t1, t2, t3)):
#     #     raise ValueError("所有输入必须为一维张量。")
#     # if not (t1.shape[0] == t2.shape[0] == t3.shape[0]):
#     #     raise ValueError("三个张量长度必须一致。")
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
        # # ipdb.set_trace()
        # for i in range(e1_down.shape[0]):
        #     plot_three_tensors_heatmap(t1=e1_down[i].mean(-1).cpu(), t2=e2_down[i].mean(-1).cpu(), t3=z[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='before_after_cov', index=i)
        #     # plot_three_tensors_heatmap(t1=e1_down[i].mean(-1).cpu(), t2=e2_down[i].mean(-1).cpu(), t3=z[i].mean(-1).cpu(), subtokens1=event_1_tokens[i], subtokens2=event_2_tokens[i], routing_name='after_cov', index=i)
        # ipdb.set_trace()


        # 5. Apply non-linearity to the combined signal
        combined_down = self.activation(combined_down)

        # 6. Up-sample back to original hidden size
        output_delta = self.up_sampler(combined_down)
        fused_output = event1_vec + output_delta
        return fused_output


class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta
class RobertaSelfAttention(nn.Module):
    def __init__(self, config, lora_config=None, use_adapter=False, use_lora=False, lora_split=False, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.use_lora = use_lora
        self.use_adapter = use_adapter
        self.lora_split = lora_split
        self.use_routing = lora_config.use_routing if lora_config is not None else False

        if use_adapter:
            # Adapter mode: use frozen standard layers
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
        else:
            # LoRA mode: create query layer based on configuration
            if self.use_lora and self.use_routing:
                if not self.lora_split:
                    # LoRA with routing (no split): use VLLinear
                    self.query = loralayer.VLLinear(
                        config.hidden_size, self.all_head_size, 
                        r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha, 
                        lora_dropout=lora_config.lora_dropout,
                        element_mul=lora_config.element_mul,
                        element_add=lora_config.element_add,
                        element_mul_expand=lora_config.element_mul_expand,
                        routing_xt=lora_config.routing_xt
                    )
                else:
                    # LoRA with routing and split: frozen layer + VLLinearSplit
                    self.query = nn.Linear(config.hidden_size, self.all_head_size)
                    for param in self.query.parameters():
                        param.requires_grad = False
                    self.lora_vl_query = loralayer.VLLinearSplit(
                        config.hidden_size, self.all_head_size,
                        r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.lora_dropout
                    )
            elif self.use_lora and not self.use_routing:
                # LoRA without routing: use standard Linear
                self.query = loralayer.Linear(
                    config.hidden_size, self.all_head_size,
                    r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout
                )
            else:
                # No LoRA: standard layer
                self.query = nn.Linear(config.hidden_size, self.all_head_size)

            # Key is always frozen
            self.key = nn.Linear(config.hidden_size, self.all_head_size)

            # Value layer based on configuration
            if self.use_lora and self.use_routing:
                if not self.lora_split:
                    # LoRA with routing (no split): use VLLinear
                    self.value = loralayer.VLLinear(
                        config.hidden_size, self.all_head_size,
                        r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.lora_dropout,
                        element_mul=lora_config.element_mul,
                        element_add=lora_config.element_add,
                        element_mul_expand=lora_config.element_mul_expand,
                        routing_xt=lora_config.routing_xt
                    )
                else:
                    # LoRA with routing and split: frozen layer + VLLinearSplit
                    self.value = nn.Linear(config.hidden_size, self.all_head_size)
                    for param in self.value.parameters():
                        param.requires_grad = False
                    self.lora_vl_value = loralayer.VLLinearSplit(
                        config.hidden_size, self.all_head_size,
                        r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.lora_dropout
                    )
            elif self.use_lora and not self.use_routing:
                # LoRA without routing: use standard Linear
                self.value = loralayer.Linear(
                    config.hidden_size, self.all_head_size,
                    r=lora_config.r_lora, lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout
                )
            else:
                # No LoRA: standard layer
                self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        vision_hidden_states: Optional[torch.Tensor] = None,
        multiply_ones: bool = False,
    ) -> tuple[torch.Tensor]:
        # Compute query layer based on configuration
        if self.use_adapter:
            mixed_query_layer = self.query(hidden_states)
        elif self.use_lora and self.use_routing:
            if not self.lora_split:
                # VLLinear forward: (x, vis_feat, multiply_ones)
                mixed_query_layer = self.query(hidden_states, vision_hidden_states, multiply_ones)
            else:
                # Frozen layer + VLLinearSplit
                mixed_query_layer = self.query(hidden_states)
                mixed_query_layer_vl = self.lora_vl_query(hidden_states, vision_hidden_states, multiply_ones=multiply_ones)
                mixed_query_layer = mixed_query_layer + mixed_query_layer_vl
        else:
            mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # Compute value layer based on configuration
            if self.use_adapter:
                value_layer = self.transpose_for_scores(self.value(hidden_states))
            elif self.use_lora and self.use_routing:
                if not self.lora_split:
                    value_layer = self.transpose_for_scores(self.value(hidden_states, vision_hidden_states, multiply_ones))
                else:
                    value_layer = self.transpose_for_scores(self.value(hidden_states))
                    value_layer_vl = self.transpose_for_scores(self.lora_vl_value(hidden_states, vision_hidden_states, multiply_ones=multiply_ones))
                    value_layer = value_layer + value_layer_vl
            else:
                value_layer = self.transpose_for_scores(self.value(hidden_states))
                

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


ROBERTA_SELF_ATTENTION_CLASSES = {
    "eager": RobertaSelfAttention,
}


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta,BERT->ROBERTA
class RobertaAttention(nn.Module):
    def __init__(self, config, lora_config=None, use_adapter=False, use_lora=False, lora_split=False, position_embedding_type=None):
        super().__init__()
        self.self = RobertaSelfAttention(
            config, lora_config=lora_config, use_adapter=use_adapter, use_lora=use_lora, 
            lora_split=lora_split, position_embedding_type=position_embedding_type
        )
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        vision_hidden_states: Optional[torch.Tensor] = None,
        multiply_ones: bool = False,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            vision_hidden_states=vision_hidden_states,
            multiply_ones=multiply_ones,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class RobertaLayer(GradientCheckpointingLayer):
    def __init__(self, config, lora_config=None, layer_idx=None, use_adapter=False, use_lora=False, lora_split=False, freeze_ln=False, adapt_pos="post_module", fusion_method="original", dataset_name=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.use_lora = use_lora
        self.lora_split = lora_split
        self.attention = RobertaAttention(config, lora_config=lora_config, use_adapter=use_adapter, use_lora=use_lora, lora_split=lora_split)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, lora_config=lora_config, use_adapter=use_adapter, use_lora=use_lora, lora_split=lora_split, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        
        # Adapter support
        self.use_adapter = use_adapter
        self.fusion_method = fusion_method
        if use_adapter and lora_config is not None:
            if fusion_method == "temporal_relation":
                self.adapter = TemporalRelationAdapter(config, lora_config, dataset_name)
            else:  # original fusion method
                self.adapter = Adapter(config, lora_config, dataset_name)
        else:
            # Freeze parameters when not using adapter (but not when using LoRA)
            if not use_lora:
                for param in self.intermediate.parameters():
                    param.requires_grad = False
                for param in self.output.parameters():
                    param.requires_grad = False
        
        # Freeze layer norms if requested
        if freeze_ln:
            for param in self.attention.output.LayerNorm.parameters():
                param.requires_grad = False
            for param in self.output.LayerNorm.parameters():
                param.requires_grad = False
        
        self.adapt_pos = adapt_pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        vision_hidden_states: Optional[torch.Tensor] = None,
        multiply_ones: bool = False,
    ) -> tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        # Apply adapter before attention if requested
        if self.use_adapter and self.adapt_pos == "pre_norm":
            hidden_states = self.adapter(hidden_states, vision_hidden_states)
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            vision_hidden_states=vision_hidden_states,
            multiply_ones=multiply_ones,
        )
        attention_output = self_attention_outputs[0]
        
        # Apply adapter after attention if requested
        if self.use_adapter and self.adapt_pos == "post_module":
            attention_output = self.adapter(attention_output, vision_hidden_states)

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply adapter before feed forward if requested
        if self.use_adapter and self.adapt_pos == "pre_norm":
            attention_output = self.adapter(attention_output, vision_hidden_states)
        
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        
        # Apply adapter after feed forward if requested
        if self.use_adapter and self.adapt_pos == "post_module":
            layer_output = self.adapter(layer_output, vision_hidden_states)
        
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class RobertaEncoder(nn.Module):
    def __init__(self, config, lora_config=None, fusion_layer=None, use_adapter=False, use_lora=False, lora_split=False, freeze_ln=False, adapt_pos="post_module", fusion_method="original", dataset_name=None):
        super().__init__()
        self.config = config
        
        # Create layers with adapter support
        if fusion_layer is None:
            fusion_layer = list(range(config.num_hidden_layers))
        
        self.layer = nn.ModuleList([
            RobertaLayer(
                config, 
                lora_config=lora_config,
                layer_idx=i,
                use_adapter=use_adapter and i in fusion_layer,
                use_lora=use_lora,
                lora_split=lora_split,
                freeze_ln=freeze_ln,
                adapt_pos=adapt_pos,
                fusion_method=fusion_method,
                dataset_name=dataset_name,
            ) for i in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        vision_hidden_states: Optional[torch.Tensor] = None,
        multiply_ones: bool = False,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                vision_hidden_states=vision_hidden_states,
                multiply_ones=multiply_ones,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@auto_docstring
class RobertaPreTrainedModel(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RobertaEmbeddings", "RobertaSelfAttention"]
    _supports_sdpa = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights with BertLMPredictionHead->RobertaLMHead
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # RobertaLMHead is not used in this implementation
        pass


# Copied from transformers.models.bert.modeling_bert.BertModel with Bert->Roberta, BERT->ROBERTA
class RobertaModel(RobertaPreTrainedModel):
    _no_split_modules = ["RobertaEmbeddings", "RobertaLayer"]

    def __init__(self, config, add_pooling_layer=True, lora_config=None, fusion_layer=None, use_adapter=False, use_lora=False, lora_split=False, freeze_ln=False, adapt_pos="post_module", fusion_method="original", dataset_name=None):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config, lora_config=lora_config, fusion_layer=fusion_layer, use_adapter=use_adapter, use_lora=use_lora, lora_split=lora_split, freeze_ln=freeze_ln, adapt_pos=adapt_pos, fusion_method=fusion_method, dataset_name=dataset_name)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_hidden_states: Optional[torch.Tensor] = None,
        multiply_ones: bool = False,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision_hidden_states=vision_hidden_states,
            multiply_ones=multiply_ones,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config, lora_config=None, fusion_layer=None, use_adapter=False, use_lora=False, lora_split=False, freeze_ln=False, adapt_pos="post_module", fusion_method="original", dataset_name=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(
            config, 
            add_pooling_layer=False,
            lora_config=lora_config,
            fusion_layer=fusion_layer,
            use_adapter=use_adapter,
            use_lora=use_lora,
            lora_split=lora_split,
            freeze_ln=freeze_ln,
            adapt_pos=adapt_pos,
            fusion_method=fusion_method,
            dataset_name=dataset_name,
        )
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_hidden_states: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision_hidden_states=vision_hidden_states,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


__all__ = [
    "RobertaForCausalLM",
    "RobertaForMaskedLM",
    "RobertaForMultipleChoice",
    "RobertaForQuestionAnswering",
    "RobertaForSequenceClassification",
    "RobertaForTokenClassification",
    "RobertaModel",
    "RobertaPreTrainedModel",
]
