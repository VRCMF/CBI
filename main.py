import os
os.environ['CURL_CA_BUNDLE'] = ''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaConfig, AutoTokenizer, AutoConfig, Qwen2Config, GPTJConfig, LlamaConfig, MistralConfig, Gemma2Config, Phi3Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import warnings
import argparse
import logging
import datetime
import sys
import time
warnings.filterwarnings('ignore')
# Import the modified model classes
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
sys.path.insert(0, models_dir)
from roberta.modeling_roberta import RobertaForSequenceClassification
from qwen2.modeling_qwen2 import Qwen2ForSequenceClassification
from llama.modeling_llama import LlamaForSequenceClassification
from mistral.modeling_mistral import MistralForSequenceClassification
from gptj.modeling_gptj import GPTJForSequenceClassification
from gemma2.modeling_gemma2 import Gemma2ForSequenceClassification
from phi3.modeling_phi3 import Phi3ForSequenceClassification
from transformers import MistralConfig, Gemma2Config, Phi3Config
import ipdb
import random
from gdllm_stage2 import run_gnn_downstream_task


# Set device (ensure GPU is used if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Note: Device info will be logged in main() after logger setup

# Dataset configurations
DATASET_CONFIGS = {
    'matres': {
        'path': 'data/temporal_relations_matres.csv',
        'labels': ['BEFORE', 'AFTER', 'SIMULTANEOUS']
    },
    'tbd': {
        'path': 'data/temporal_relations_tbd.csv',
        'labels': ['BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS']
    },
    'tcr': {
        'path': 'data/temporal_relations_tcr.csv',
        'labels': ['BEFORE', 'AFTER', 'SIMULTANEOUS']
    },
    'tdd_auto': {
        'path': 'data/temporal_relations_tdd_auto.csv',
        'labels': ['BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS']
    },
    'tdd_man': {
        'path': 'data/temporal_relations_tdd_man.csv',
        'labels': ['BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS']
    }
}

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(output_dir):
    """Setup logging to both console and file"""
    # Create log directory if it doesn't exist
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("TEMPORAL RELATION CLASSIFICATION TRAINING LOG")
    logger.info("="*60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")
    
    # Setup exception handling
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupt to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the exception
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        logger.critical("Program terminated due to unexpected error!")
        logger.critical("="*60)
    
    # Install the exception handler
    sys.excepthook = handle_exception
    
    return logger

class TemporalRelationDataset(Dataset):
    """
    仅返回原始文本与标签；不在 __getitem__ 里做 padding/tokenization。
    """
    def __init__(self, data: pd.DataFrame, tokenizer, dataset_name: str,
                 max_length: int = 512, use_routing: bool = True):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_routing = use_routing

        # labels / label_map
        self.labels = DATASET_CONFIGS[dataset_name]["labels"]
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}
        self.num_labels = len(self.label_map)

        # 兜底 pad_token_id
        if self.tokenizer.pad_token_id is None:
            # 一些 LLM（如 GPT2-like）缺省没有 pad token；这里设置为 eos_token
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # 退一步新增一个 [PAD]
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        text1 = str(row["event_ctx_1"]).strip()
        text2 = str(row["event_ctx_2"]).strip()
        label = self.label_map[str(row["relation"]).strip()]
        if self.use_routing:
            return {"text1": text1, "text2": text2, "labels": label}
        else:
            # 非 routing：如果你想把两段合成一路输入，这里先只返回文本，collate 再拼接
            return {"text1": text1, "text2": text2, "labels": label}


def collate_fn_pair_equal(
    batch,
    tokenizer,
    max_length: int = 512,
    pad_to_multiple_of: int = 8,
    use_routing: bool = True,
    sep_token: str = "[SEP]"
):
    """
    目标：
      1) 对每个样本，event1 和 event2 一起做 tokenization（不 padding）；
      2) 先将这对序列 pad 到“二者中较长者”的长度（逐样本配对等长）；
      3) 再跨样本把每一路 pad 到本 batch 最大长度（得到规则张量）；
    可选：将跨样本维度 pad 到 8/16 的倍数，利好 Tensor Cores。
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # --- 工具函数 ---
    def _pad_list(seq, tgt_len, pad_val):
        return seq + [pad_val] * (tgt_len - len(seq))

    def _pad_batch(list_of_lists, pad_val):
        lens = [len(x) for x in list_of_lists]
        max_len = max(lens) if lens else 0

        # 可选对齐到 8/16 倍数
        if pad_to_multiple_of is not None and max_len % pad_to_multiple_of != 0:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        out = torch.full((len(list_of_lists), max_len), pad_val, dtype=torch.long)
        for i, x in enumerate(list_of_lists):
            out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        return out

    # --- 收集逐样本配对后的结果 ---
    pair_ids_1, pair_masks_1 = [], []
    pair_ids_2, pair_masks_2 = [], []
    labels = []

    for b in batch:
        t1 = b["text1"]
        t2 = b["text2"]
        labels.append(b["labels"])

        if use_routing:
            # 一起做 tokenizer（两条序列），不 padding
            enc = tokenizer(
                [t1, t2],
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            )
            ids1, ids2 = enc["input_ids"]
            mask1, mask2 = enc["attention_mask"]
        else:
            # 非 routing：若你想把两段文本拼成一路传给模型（例如 "t1 [SEP] t2"）
            # 这里示范把两段文本各自编码，再在 id 层面拼接一个 sep_token
            # 也可以直接 tokenizer(f"{t1} {sep_token} {t2}", ...) ——取决于你的模型约定
            enc1 = tokenizer(t1, truncation=True, max_length=max_length, add_special_tokens=True)
            enc2 = tokenizer(t2, truncation=True, max_length=max_length, add_special_tokens=True)

            # 将 sep_token 以 id 形式插入中间
            if sep_token not in tokenizer.get_vocab():
                # 若不存在该 token，动态添加
                tokenizer.add_tokens([sep_token])
            sep_id = tokenizer.convert_tokens_to_ids(sep_token)

            ids1 = enc1["input_ids"]
            mask1 = enc1["attention_mask"]
            ids2 = enc2["input_ids"]
            mask2 = enc2["attention_mask"]

            # 拼接方案（保持“成对等长”的需求：仍当作两路来对齐）
            # 如果你确实要单路输入，建议改写下游模型接收单路；此处保留两路以满足你的“配对等长”要求
            # 也可以把 ids1 + [sep_id] + ids2 合并为一路，这里提供注释版：
            # merged_ids  = ids1 + [sep_id] + ids2
            # merged_mask = mask1 + [1] + mask2
            # 然后就只保留一路。不过你当前下游接口是两路，这里仍按两路处理。

        # 逐样本的目标长度：两者中较长者
        
        m = max(len(ids1), len(ids2))

        # 逐样本配对等长
        ids1  = _pad_list(ids1,  m, pad_id)
        ids2  = _pad_list(ids2,  m, pad_id)
        mask1 = _pad_list(mask1, m, 0)
        mask2 = _pad_list(mask2, m, 0)

        pair_ids_1.append(ids1)
        pair_masks_1.append(mask1)
        pair_ids_2.append(ids2)
        pair_masks_2.append(mask2)

    # 跨样本对齐
    input_ids             = _pad_batch(pair_ids_1, pad_id)
    attention_mask        = _pad_batch(pair_masks_1, 0)
    vision_input_ids      = _pad_batch(pair_ids_2, pad_id)
    vision_attention_mask = _pad_batch(pair_masks_2, 0)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "vision_input_ids": vision_input_ids,
        "vision_attention_mask": vision_attention_mask,
        "labels": labels
    }

def collate_fn_single_merged(
    batch,
    tokenizer,
    max_length: int = 512,
    pad_to_multiple_of: int = 8,
    sep_token: str = "[SEP]"
):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    pad_id = tokenizer.pad_token_id

    texts = [f"{b['text1']} {sep_token} {b['text2']}" for b in batch]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt"
    )

    # 可选对齐到倍数
    if pad_to_multiple_of is not None:
        L = enc["input_ids"].size(1)
        if L % pad_to_multiple_of != 0:
            newL = ((L + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            pad_len = newL - L
            enc["input_ids"] = torch.nn.functional.pad(enc["input_ids"], (0, pad_len), value=pad_id)
            enc["attention_mask"] = torch.nn.functional.pad(enc["attention_mask"], (0, pad_len), value=0)

    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    enc["labels"] = labels
    return enc

def format_memory_size(size_bytes):
    """Format memory size in bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def load_and_split_data(dataset_name):
    """Load data and split based on the split column"""
    print("Loading data...")
    data_path = DATASET_CONFIGS[dataset_name]['path']
    df = pd.read_csv(data_path)
    
    # Ensure only valid relations are included
    valid_relations = DATASET_CONFIGS[dataset_name]['labels']
    df = df[df['relation'].isin(valid_relations)].reset_index(drop=True)
    
    print(f"Total samples: {len(df)}")
    print("Relation distribution:")
    print(df['relation'].value_counts())
    print("\nSplit distribution:")
    print(df['split'].value_counts())
    
    # Split data based on the split column
    train_data = df[df['split'] == 'train'].reset_index(drop=True)
    val_data = df[df['split'] == 'validation'].reset_index(drop=True)  # Changed from 'val' to 'validation'
    test_data = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"\nTrain: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    if len(val_data) == 0:
        raise ValueError("No validation data found! Please ensure the dataset has a 'validation' split.")
    
    return train_data, val_data, test_data

class TemporalRelationClassifier(nn.Module):
    """Temporal relation classifier with adapter capabilities"""
    def __init__(self, model_name='roberta-base', num_labels=4, lora_config=None, fusion_layer=None, use_adapter=False, use_lora=False, lora_split=False, freeze_ln=False, adapt_pos="post_module", fusion_method="original", logger=None, dataset_name=None):
        super().__init__()
        
        # Determine model type based on model name
        if 'roberta' in model_name.lower():
            self.model_type = 'roberta'
            try:
                self.config = RobertaConfig.from_pretrained(model_name, local_files_only=True)
            except OSError:
                self.config = RobertaConfig.from_pretrained(model_name)
            try:
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name, local_files_only=True)
                print(f"Loaded tokenizer from local cache for {model_name}")
            except OSError:
                print(f"Local cache not found, downloading tokenizer for {model_name}")
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        elif 'mistral' in model_name.lower():
            self.model_type = 'mistral'
            try:
                self.config = MistralConfig.from_pretrained(model_name, local_files_only=True)
            except OSError:
                self.config = MistralConfig.from_pretrained(model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                print(f"Loaded tokenizer from local cache for {model_name}")
            except OSError:
                print(f"Local cache not found, downloading tokenizer for {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.pad_token_id = self.tokenizer.eos_token_id
        elif 'llama' in model_name.lower() or 'meta-llama' in model_name.lower():
            self.model_type = 'llama'
            print(f"Detected LLaMA model: {model_name}")
            try:
                self.config = LlamaConfig.from_pretrained(model_name, local_files_only=True)
            except OSError:
                self.config = LlamaConfig.from_pretrained(model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                print(f"Loaded tokenizer from local cache for {model_name}")
            except OSError:
                print(f"Local cache not found, downloading tokenizer for {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.config.pad_token_id = self.tokenizer.pad_token_id
            # For LLaMA models, we might need to set chat template
            if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                if logger:
                    logger.warning("LLaMA tokenizer missing chat_template, setting a simple one")
                if hasattr(self.tokenizer, 'chat_template'):
                    self.tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        elif 'qwen' in model_name.lower():
            self.model_type = 'qwen'
            if 'qwen2.5' in model_name.lower():
                print(f"Detected Qwen2.5 model: {model_name}")
                try:
                    self.config = Qwen2Config.from_pretrained(model_name, local_files_only=True)
                except OSError:
                    self.config = Qwen2Config.from_pretrained(model_name)
            elif 'qwen2' in model_name.lower():
                print(f"Detected Qwen2 model: {model_name}")
                try:
                    self.config = Qwen2Config.from_pretrained(model_name, local_files_only=True)
                except OSError:
                    self.config = Qwen2Config.from_pretrained(model_name)
            else:
                print(f"Detected Qwen model: {model_name}")
                try:
                    self.config = AutoConfig.from_pretrained(model_name, local_files_only=True)
                except OSError:
                    self.config = AutoConfig.from_pretrained(model_name)
            try:
                # Try offline first
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                print(f"Loaded tokenizer from local cache for {model_name}")
            except OSError:
                # Fall back to online if local files not available
                print(f"Local cache not found, downloading tokenizer for {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.pad_token_id = self.tokenizer.eos_token_id
        elif 'gpt-j' in model_name.lower() or 'gptj' in model_name.lower():
            self.model_type = 'gptj'
            print(f"Detected GPT-J model: {model_name}")
            try:
                self.config = GPTJConfig.from_pretrained(model_name, local_files_only=True)
            except OSError:
                self.config = GPTJConfig.from_pretrained(model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                print(f"Loaded tokenizer from local cache for {model_name}")
            except OSError:
                print(f"Local cache not found, downloading tokenizer for {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.pad_token_id = self.tokenizer.eos_token_id
        elif 'gemma' in model_name.lower():
            self.model_type = 'gemma2'
            print(f"Detected Gemma2 model: {model_name}")
            try:
                self.config = Gemma2Config.from_pretrained(model_name, local_files_only=True)
            except OSError:
                self.config = Gemma2Config.from_pretrained(model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                print(f"Loaded tokenizer from local cache for {model_name}")
            except OSError:
                print(f"Local cache not found, downloading tokenizer for {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.pad_token_id = self.tokenizer.eos_token_id
        elif 'phi' in model_name.lower():
            self.model_type = 'phi3'
            print(f"Detected Phi3 model: {model_name}")
            try:
                self.config = Phi3Config.from_pretrained(model_name, local_files_only=True)
            except OSError:
                self.config = Phi3Config.from_pretrained(model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                print(f"Loaded tokenizer from local cache for {model_name}")
            except OSError:
                print(f"Local cache not found, downloading tokenizer for {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        self.config.num_labels = num_labels
        self.use_routing = lora_config.use_routing if lora_config is not None else False
        
        if logger:
            logger.info(f"Using model type: {self.model_type}")
        
        # Initialize the appropriate model based on type
        if self.model_type == 'roberta':
            self.model = RobertaForSequenceClassification(
                self.config,
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
        elif self.model_type == 'mistral':
            self.model = MistralForSequenceClassification(
                self.config,
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
        elif self.model_type == 'llama':
            self.model = LlamaForSequenceClassification(
                self.config,
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
        elif self.model_type == 'qwen':
            self.model = Qwen2ForSequenceClassification(
                self.config,
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
        elif self.model_type == 'gptj':
            self.model = GPTJForSequenceClassification(
                self.config,
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
        elif self.model_type == 'gemma2':
            self.model = Gemma2ForSequenceClassification(
                self.config,
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
        elif self.model_type == 'phi3':
            self.model = Phi3ForSequenceClassification(
                self.config,
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
        # Always freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # If using adapter or LoRA, unfreeze only relevant parameters
        if use_lora:
            lora_params_found = 0
            for name, param in self.model.named_parameters():
                # More comprehensive LoRA/adapter parameter detection
                if any(lora_keyword in name.lower() for lora_keyword in ['lora_a', 'lora_b', 'lora_dropout', 'lora_', 'routing', 'adapter', 'conv']):
                    if 'k_proj' not in name.lower():
                        param.requires_grad = True
                        lora_params_found += 1
                        print(f"Unfreezing LoRA/adapter parameter: {name}")
                        if logger:
                            logger.debug(f"Unfreezing LoRA/adapter parameter: {name}")
        elif use_adapter:
            lora_params_found = 0
            for name, param in self.model.named_parameters():
                if any(lora_keyword in name.lower() for lora_keyword in ['adapter', 'routing']):
                    param.requires_grad = True
                    lora_params_found += 1
                    print(f"Unfreezing LoRA/adapter parameter: {name}")
                    if logger:
                        logger.debug(f"Unfreezing LoRA/adapter parameter: {name}")
            
            # if lora_params_found == 0:
            #     if logger:
            #         logger.warning("No LoRA/adapter parameters found! This might cause training issues.")
            #         logger.info("Available parameter names:")
            #         print(f"Unfreezing LoRA parameter: {name}")
            #         for name, param in self.model.named_parameters():
            #             if 'lora' in name.lower() or 'routing' in name.lower() or 'adapter' in name.lower() or 'conv' in name.lower():
            #                 logger.info(f"  - {name}")
            # else:
            #     if logger:
            #         logger.info(f"Found and unfroze {lora_params_found} LoRA/adapter parameter tensors")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if logger:
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            if trainable_params > 0:
                logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
            else:
                logger.warning("No trainable parameters found!")
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, vision_input_ids=None, vision_attention_mask=None, labels=None):
        if self.use_routing and vision_input_ids is not None:
            # Get embeddings for vision_input_ids based on model type
            if self.model_type == 'roberta':
                vision_embeds = self.model.roberta.embeddings(
                    input_ids=vision_input_ids.unsqueeze(0) if vision_input_ids.dim()==1 else vision_input_ids
                )
            elif self.model_type == 'mistral':
                vision_embeds = self.model.model.embed_tokens(
                    vision_input_ids.unsqueeze(0) if vision_input_ids.dim()==1 else vision_input_ids
                )
            elif self.model_type == 'llama':
                vision_embeds = self.model.model.embed_tokens(
                    vision_input_ids.unsqueeze(0) if vision_input_ids.dim()==1 else vision_input_ids
                )
            elif self.model_type == 'qwen':
                vision_embeds = self.model.model.embed_tokens(
                    vision_input_ids.unsqueeze(0) if vision_input_ids.dim()==1 else vision_input_ids
                )
            elif self.model_type == 'gptj':
                vision_embeds = self.model.transformer.wte(
                    vision_input_ids.unsqueeze(0) if vision_input_ids.dim()==1 else vision_input_ids
                )
            elif self.model_type == 'gemma2':
                vision_embeds = self.model.model.embed_tokens(
                    vision_input_ids.unsqueeze(0) if vision_input_ids.dim()==1 else vision_input_ids
                )
            elif self.model_type == 'phi3':
                vision_embeds = self.model.model.embed_tokens(
                    vision_input_ids.unsqueeze(0) if vision_input_ids.dim()==1 else vision_input_ids
                )
            else:
                raise ValueError(f"Unsupported model type for routing: {self.model_type}")
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_hidden_states=vision_embeds,
                labels=labels
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        logits = outputs.logits
        return logits

def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=2e-5, 
                output_dir='./', logger=None, gradient_accumulation_steps=1, dataset_name=None):
    """Train the model with gradient accumulation support"""

    model.to(device)

    # Only pass trainable parameters to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Make sure LoRA parameters are properly unfrozen.")
    
    if logger:
        logger.info(f"Optimizer will train {len(trainable_params)} parameter tensors")
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        # Start timing and reset memory stats
        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if 'vision_input_ids' in batch:
                vision_input_ids = batch['vision_input_ids'].to(device)
                vision_attention_mask = batch['vision_attention_mask'].to(device)
                logits = model(input_ids, attention_mask, vision_input_ids, vision_attention_mask, labels)
            else:
                logits = model(input_ids, attention_mask, labels)

            loss = criterion(logits, labels)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                if logger:
                    logger.error(f"Invalid loss detected at epoch {epoch+1}: {loss.item()}")
                    logger.error("This might be due to:\n- Learning rate too high\n- Gradient explosion\n- Numerical instability")
                raise ValueError(f"Training failed due to invalid loss: {loss.item()}")

            loss.backward()

            # Gradient accumulation update
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                # if total_norm > 10.0 and logger:
                #     logger.warning(f"Large gradient norm detected: {total_norm:.2f}")
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps  # rescale back for logging
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch time and memory
        epoch_time = time.time() - epoch_start_time
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                if 'vision_input_ids' in batch:
                    vision_input_ids = batch['vision_input_ids'].to(device)
                    vision_attention_mask = batch['vision_attention_mask'].to(device)
                    logits = model(input_ids, attention_mask, vision_input_ids, vision_attention_mask, labels)
                else:
                    logits = model(input_ids, attention_mask)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate F1 scores
        train_f1_micro = f1_score(train_labels, train_preds, average='micro')
        train_f1_macro = f1_score(train_labels, train_preds, average='macro')
        val_f1_micro = f1_score(val_labels, val_preds, average='micro')
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')

        
        
        # Get memory usage
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()
            memory_info = f"Peak GPU Memory: {format_memory_size(peak_memory)}, Current GPU Memory: {format_memory_size(current_memory)}"
        else:
            memory_info = "GPU not available"

        if logger:
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
            logger.info(f"Memory Usage: {memory_info}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train F1 (micro/macro): {train_f1_micro:.4f}/{train_f1_macro:.4f}")
            logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}, Val F1 (micro/macro): {val_f1_micro:.4f}/{val_f1_macro:.4f}")
            logger.info("-" * 50)
        # ipdb.set_trace()
        # Save best model
        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            torch.save(model.state_dict(), f'{output_dir}/best_temporal_model_{dataset_name}.pt')
    
    return model

def evaluate_model(model, test_loader, dataset_name, logger=None):
    """Evaluate the model on test set"""
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if 'vision_input_ids' in batch:
                vision_input_ids = batch['vision_input_ids'].to(device)
                vision_attention_mask = batch['vision_attention_mask'].to(device)
                logits = model(input_ids, attention_mask, vision_input_ids, vision_attention_mask, None)
            else:
                logits = model(input_ids, attention_mask, None)
            
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate F1 scores
    f1_micro = f1_score(test_labels, test_preds, average='micro')
    f1_macro = f1_score(test_labels, test_preds, average='macro')
    
    # Get label names for the specific dataset
    label_names = DATASET_CONFIGS[dataset_name]['labels']
    report = classification_report(test_labels, test_preds, target_names=label_names, digits=4)
    
    if logger:
        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Micro-F1: {f1_micro:.4f}")
        logger.info(f"Macro-F1: {f1_macro:.4f}")
        logger.info("Detailed Classification Report:")
        logger.info(f"\n{report}")
    
    return f1_micro, f1_macro, report

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Temporal Relation Classification with Adapters')
    parser.add_argument('--dataset', type=str, default='matres', choices=list(DATASET_CONFIGS.keys()),
                      help='Dataset to use (matres, tbd, tcr, tdd_auto, tdd_man)')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                      help='Model name (e.g., roberta-base, roberta-large, '
                           'meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.1-70B, meta-llama/Llama-3-8B, meta-llama/Llama-3-70B, '
                           'meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf, '
                           'mistralai/Mistral-7B-v0.1, '
                           'Qwen/Qwen2.5-7B, Qwen/Qwen2.5-14B, Qwen/Qwen2-7B, '
                           'EleutherAI/gpt-j-6b, '
                           'google/gemma-2b, google/gemma-7b, '
                           'microsoft/Phi-3.5-mini-instruct, microsoft/Phi-3-small-8k-instruct, microsoft/Phi-3-medium-4k-instruct)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training and evaluation')
    parser.add_argument('--r_lora', type=int, default=128, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=128, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--use_routing', type=str, default='False', help='Use routing (True/False)')
    parser.add_argument('--add_relu', type=str, default='False', help='Add ReLU (True/False)')
    parser.add_argument('--element_mul', type=str, default='True', help='Element multiplication (True/False)')
    parser.add_argument('--element_add', type=str, default='False', help='Element addition (True/False)')
    parser.add_argument('--element_mul_expand', type=str, default='False', help='Element multiplication expand (True/False)')
    parser.add_argument('--element_minus', type=str, default='False', help='Element minus (True/False)')
    parser.add_argument('--routing_xt', type=str, default='False', help='Routing XT (True/False)')
    parser.add_argument('--conv_kernel_size', type=int, default=5, help='Convolution kernel size')
    parser.add_argument('--fusion_method', type=str, default='original', 
                      choices=['original', 'temporal_relation', 'temporal_constrain', 'neopy_inspired', 'contextual_temporal', 'unified_temporal', 'point_based', 'temporal_reasoning', 'relational_gated_hybrid', 'factorized_bilinear', 'poly_interactive_gated', 'gated_cross_attention', 'tensor_fusion'],
                      help='Fusion method for adapters')
    parser.add_argument('--use_lora', type=str, default='False', help='Use LoRA (True/False)')
    parser.add_argument('--lora_split', type=str, default='False', help='Use LoRA split (True/False)')
    parser.add_argument('--use_adapter', type=str, default='False', help='Use adapter (True/False)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for training')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory for results')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    # Setup logging first
    logger = setup_logging(args.output_dir)
    

    logger.info("Training Arguments:")
    logger.info("-" * 40)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("-" * 40)
    
    # Convert string arguments to boolean
    use_routing = args.use_routing.lower() == 'true'
    add_relu = args.add_relu.lower() == 'true'
    element_mul = args.element_mul.lower() == 'true'
    element_add = args.element_add.lower() == 'true'
    element_mul_expand = args.element_mul_expand.lower() == 'true'
    element_minus = args.element_minus.lower() == 'true'
    routing_xt = args.routing_xt.lower() == 'true'
    use_lora = args.use_lora.lower() == 'true'
    lora_split = args.lora_split.lower() == 'true'
    use_adapter = args.use_adapter.lower() == 'true'
    conv_kernel_size = args.conv_kernel_size
    
    # Log device information
    logger.info(f"Using device: {device}")
    
    # 1. Load and split data
    train_data, val_data, test_data = load_and_split_data(args.dataset)
    if 'roberta' in args.model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    elif 'bert' in args.model_name.lower():
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    elif 'mistral' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif 'llama' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif 'qwen' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif 'gpt' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif 'gemma' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif 'phi' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # 2. Create LoRA config
    class LoRAConfig:
        def __init__(self):
            self.r_lora = args.r_lora
            self.lora_alpha = args.lora_alpha
            self.lora_dropout = args.lora_dropout
            self.use_routing = use_routing
            self.add_relu = add_relu
            self.element_mul = element_mul
            self.element_add = element_add
            self.element_mul_expand = element_mul_expand
            self.element_minus = element_minus
            self.routing_xt = routing_xt
            self.conv_kernel_size = conv_kernel_size

    
    lora_config = LoRAConfig()

    # 3. Create datasets
    print("Creating datasets...")
    use_routing = lora_config.use_routing
    train_dataset = TemporalRelationDataset(train_data, tokenizer, args.dataset, use_routing=use_routing)
    val_dataset = TemporalRelationDataset(val_data, tokenizer, args.dataset, use_routing=use_routing)
    test_dataset = TemporalRelationDataset(test_data, tokenizer, args.dataset, use_routing=use_routing)
    

    # 3. Initialize model with adapter
    logger.info(f"Initializing model with adapter: {args.model_name}")
    logger.info(f"Using conv kernel size: {conv_kernel_size}")
    # Get number of labels for the selected dataset
    num_labels = len(DATASET_CONFIGS[args.dataset]['labels'])
    print(f"Number of labels for dataset {args.dataset}: {num_labels}")
    
    # 4. Create data loaders
    print(f"Creating data loaders with batch size: {args.batch_size}")
    if use_routing:
        collate_routing = lambda batch: collate_fn_pair_equal(
            batch,
            tokenizer=tokenizer,
            max_length=128,
            pad_to_multiple_of=8,
            use_routing=True,        # 两路：event1 & event2 各一路
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_routing)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_routing)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_routing)
    else:
        collate_single = lambda batch: collate_fn_single_merged(
            batch,
            tokenizer=tokenizer,
            max_length=256,
            pad_to_multiple_of=8,
            sep_token="[SEP]"
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_single)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_single)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_single)
    
    # 6. Initialize model with adapter
    print("Initializing model with adapter...")
    # Set fusion layers based on model type
    if 'roberta-large' in args.model_name:
        fusion_layers = list(range(24))  # roberta-large has 24 layers
    elif 'roberta' in args.model_name:
        fusion_layers = list(range(12))  # roberta-base has 12 layers
    elif 'mistral' in args.model_name.lower():
        fusion_layers = list(range(32))  # mistral-7b has 32 layers
    elif 'llama' in args.model_name.lower() or 'meta-llama' in args.model_name.lower():
        if 'llama-3.1' in args.model_name.lower():
            if '8b' in args.model_name.lower():
                fusion_layers = list(range(32))  # llama-3.1-8b has 32 layers
            elif '70b' in args.model_name.lower():
                fusion_layers = list(range(80))  # llama-3.1-70b has 80 layers
            elif '405b' in args.model_name.lower():
                fusion_layers = list(range(126))  # llama-3.1-405b has 126 layers
            else:
                fusion_layers = list(range(32))  # default for llama-3.1
        elif 'llama-3' in args.model_name.lower() or 'llama3' in args.model_name.lower():
            if '8b' in args.model_name.lower():
                fusion_layers = list(range(32))  # llama-3-8b has 32 layers
            elif '70b' in args.model_name.lower():
                fusion_layers = list(range(80))  # llama-3-70b has 80 layers
            else:
                fusion_layers = list(range(32))  # default for llama-3
        elif 'llama-2' in args.model_name.lower() or 'llama2' in args.model_name.lower():
            if '7b' in args.model_name.lower():
                fusion_layers = list(range(32))  # llama-2-7b has 32 layers
            elif '13b' in args.model_name.lower():
                fusion_layers = list(range(40))  # llama-2-13b has 40 layers
            elif '70b' in args.model_name.lower():
                fusion_layers = list(range(80))  # llama-2-70b has 80 layers
            else:
                fusion_layers = list(range(32))  # default for llama-2
        elif '7b' in args.model_name.lower():
            fusion_layers = list(range(32))  # llama-7b has 32 layers
        elif '8b' in args.model_name.lower():
            fusion_layers = list(range(32))  # llama-8b has 32 layers
        elif '13b' in args.model_name.lower():
            fusion_layers = list(range(40))  # llama-13b has 40 layers
        elif '70b' in args.model_name.lower():
            fusion_layers = list(range(80))  # llama-70b has 80 layers
        else:
            fusion_layers = list(range(32))  # default to 32 layers
    elif 'qwen' in args.model_name.lower():
        if 'qwen2.5' in args.model_name.lower():
            if '0.5b' in args.model_name.lower():
                fusion_layers = list(range(24))  # qwen2.5-0.5b has 24 layers
            elif '1.5b' in args.model_name.lower():
                fusion_layers = list(range(28))  # qwen2.5-1.5b has 28 layers
            elif '3b' in args.model_name.lower():
                fusion_layers = list(range(36))  # qwen2.5-3b has 36 layers
            elif '7b' in args.model_name.lower():
                fusion_layers = list(range(28))  # qwen2.5-7b has 28 layers
            elif '14b' in args.model_name.lower():
                fusion_layers = list(range(40))  # qwen2.5-14b has 40 layers
            elif '32b' in args.model_name.lower():
                fusion_layers = list(range(64))  # qwen2.5-32b has 64 layers
            elif '72b' in args.model_name.lower():
                fusion_layers = list(range(80))  # qwen2.5-72b has 80 layers
            else:
                fusion_layers = list(range(28))  # default to 28 layers for qwen2.5
        elif 'qwen2' in args.model_name.lower():
            if '0.5b' in args.model_name.lower():
                fusion_layers = list(range(24))  # qwen2-0.5b has 24 layers
            elif '1.5b' in args.model_name.lower():
                fusion_layers = list(range(28))  # qwen2-1.5b has 28 layers
            elif '7b' in args.model_name.lower():
                fusion_layers = list(range(32))  # qwen2-7b has 32 layers
            elif '72b' in args.model_name.lower():
                fusion_layers = list(range(80))  # qwen2-72b has 80 layers
            else:
                fusion_layers = list(range(32))  # default qwen2 layers
        else:
            fusion_layers = list(range(32))  # default qwen layers
    elif 'gpt-j' in args.model_name.lower() or 'gptj' in args.model_name.lower():
        fusion_layers = list(range(28))  # gpt-j-6b has 28 layers
    elif 'gemma' in args.model_name.lower():
        if '2b' in args.model_name.lower():
            fusion_layers = list(range(18))  # gemma-2b has 18 layers
        elif '7b' in args.model_name.lower():
            fusion_layers = list(range(28))  # gemma-7b has 28 layers
        elif '9b' in args.model_name.lower():
            fusion_layers = list(range(42))  # gemma-9b has 42 layers
        elif '27b' in args.model_name.lower():
            fusion_layers = list(range(46))  # gemma-27b has 46 layers
        else:
            fusion_layers = list(range(28))  # default to 28 layers for gemma
    elif 'phi' in args.model_name.lower():
        if 'phi-3.5-mini' in args.model_name.lower():
            fusion_layers = list(range(32))  # phi-3.5-mini has 32 layers
        elif 'phi-3-small' in args.model_name.lower():
            fusion_layers = list(range(32))  # phi-3-small has 32 layers
        elif 'phi-3-medium' in args.model_name.lower():
            fusion_layers = list(range(40))  # phi-3-medium has 40 layers
        elif 'phi-3' in args.model_name.lower():
            fusion_layers = list(range(32))  # default phi-3 has 32 layers
        else:
            fusion_layers = list(range(32))  # default to 32 layers for phi
    else:
        fusion_layers = list(range(12))  # default fallback
    
    logger.info(f"Using {len(fusion_layers)} fusion layers: {fusion_layers}")
        
    model = TemporalRelationClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        lora_config=lora_config,
        fusion_layer=fusion_layers,  # Use all layers
        use_adapter=use_adapter,  # Use adapter only when not using LoRA
        use_lora=use_lora,
        lora_split=lora_split,
        freeze_ln=True,
        adapt_pos="post_module",
        fusion_method=args.fusion_method,
        logger=logger,
        dataset_name=args.dataset
    )
    
    # # 4. Create datasets using the model's tokenizer
    # logger.info("Creating datasets...")
    # use_routing = lora_config.use_routing
    # train_dataset = TemporalRelationDataset(train_data, model.tokenizer, args.dataset, use_routing=use_routing)
    # val_dataset = TemporalRelationDataset(val_data, model.tokenizer, args.dataset, use_routing=use_routing)
    # test_dataset = TemporalRelationDataset(test_data, model.tokenizer, args.dataset, use_routing=use_routing)
    
    # # 5. Create data loaders
    # logger.info(f"Creating data loaders with batch size: {args.batch_size}")
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 6. Train model
    logger.info("Starting training...")
    model = train_model(model, train_loader, val_loader, num_epochs=args.num_epochs, learning_rate=args.learning_rate, output_dir=args.output_dir, logger=logger, gradient_accumulation_steps=args.gradient_accumulation_steps, dataset_name=args.dataset)
    
    # 7. Load best model and evaluate
    logger.info("Loading best model for evaluation...")
    model.load_state_dict(torch.load(f'{args.output_dir}/best_temporal_model_{args.dataset}.pt'))
    
    # 8. Evaluate on test set
    f1_micro, f1_macro, report = evaluate_model(model, test_loader, args.dataset, logger)

    # =====================================================================================
    # START: Call the GNN function here
    # # =====================================================================================
    # # 10. Run the GNN downstream task using the best LLM as a feature extractor
    
    # # We need the original dataframes again for graph construction
    # # train_data, val_data, test_data = load_and_split_data(args.dataset)
    
    # # The 'model' variable here is the best LLM loaded from the checkpoint
    # run_gnn_downstream_task(
    #     llm_model=model,
    #     llm_tokenizer=tokenizer,
    #     train_df=train_data,
    #     val_df=val_data,
    #     test_df=test_data,
    #     dataset_name=args.dataset,
    #     args=args,
    #     device=device,
    #     logger=logger,
    # )
    # # =====================================================================================
    # # END: Call the GNN function here
    # # =====================================================================================
    
    # 9. Save results
    with open(f'{args.output_dir}/temporal_relation_results.txt', 'w') as f:
        f.write(f"TEMPORAL RELATION CLASSIFICATION RESULTS - Dataset: {args.dataset}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Micro-F1: {f1_micro:.4f}\n")
        f.write(f"Macro-F1: {f1_macro:.4f}\n")
        f.write("\nDetailed Classification Report:\n")
        f.write(report)
    
    logger.info(f"Results saved to '{args.output_dir}/temporal_relation_results.txt'")
    logger.info(f"Final Test Micro-F1: {f1_micro:.4f}")
    logger.info(f"Final Test Macro-F1: {f1_macro:.4f}")
    
    # 10. Clean up: Delete the saved model to save disk space
    model_path = f'{args.output_dir}/best_temporal_model_{args.dataset}.pt'
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Cleaned up: Deleted model file '{model_path}' to save disk space")
        else:
            logger.info(f"Model file '{model_path}' not found, no cleanup needed")
    except Exception as e:
        logger.warning(f"Could not delete model file '{model_path}': {e}")
        
    logger.info("Training and evaluation completed successfully!")
    logger.info("="*60)

if __name__ == "__main__":
    main()  