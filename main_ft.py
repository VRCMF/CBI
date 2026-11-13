import os
os.environ['CURL_CA_BUNDLE'] = ''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score
import warnings
import argparse
import logging
import datetime
import sys
import time
import functools
warnings.filterwarnings('ignore')
import random


def setup_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_dist():
    dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

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

def build_model(model_name, num_labels, logger=None):
    """Load a standard HuggingFace ForSequenceClassification model."""
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    # Ensure pad_token_id is set for decoder-only models
    if config.pad_token_id is None and hasattr(config, 'eos_token_id'):
        config.pad_token_id = config.eos_token_id if isinstance(config.eos_token_id, int) \
            else config.eos_token_id[0]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process() and logger:
        logger.info(f"Total parameters: {total_params:,}")

    return model


def get_transformer_layer_cls(model):
    """Return the decoder/encoder layer class for FSDP auto-wrapping."""
    # Covers LLaMA, Mistral, Qwen2, Gemma, Phi, GPT-J, RoBERTa, etc.
    import transformers
    candidates = [
        "LlamaDecoderLayer", "MistralDecoderLayer", "Qwen2DecoderLayer",
        "Gemma2DecoderLayer", "Phi3DecoderLayer", "GPTJBlock",
        "RobertaLayer", "BertLayer",
    ]
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is not None and any(isinstance(m, cls) for m in model.modules()):
            return cls
    # fallback: no wrapping hint
    return None

def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=2e-5,
                output_dir='./', logger=None, gradient_accumulation_steps=1, dataset_name=None):
    """Train the FSDP-wrapped model."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []
        optimizer.zero_grad()

        train_loader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            labels = batch['labels'].to(local_rank)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                if logger and is_main_process():
                    logger.error(f"Invalid loss at epoch {epoch+1}: {loss.item()}")
                raise ValueError(f"Training failed due to invalid loss: {loss.item()}")

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps
            preds = torch.argmax(outputs.logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

        # ---- Validation (only on rank 0 to avoid duplicates) ----
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(local_rank)
                attention_mask = batch['attention_mask'].to(local_rank)
                labels = batch['labels'].to(local_rank)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        epoch_time = time.time() - epoch_start_time
        train_f1_micro = f1_score(train_labels_list, train_preds, average='micro')
        train_f1_macro = f1_score(train_labels_list, train_preds, average='macro')
        val_f1_micro = f1_score(val_labels_list, val_preds, average='micro')
        val_f1_macro = f1_score(val_labels_list, val_preds, average='macro')

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_info = f"Peak GPU Memory: {format_memory_size(peak_memory)}"
        else:
            memory_info = "GPU not available"

        if is_main_process() and logger:
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
            logger.info(f"Memory Usage: {memory_info}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train F1 (micro/macro): {train_f1_micro:.4f}/{train_f1_macro:.4f}")
            logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}, Val F1 (micro/macro): {val_f1_micro:.4f}/{val_f1_macro:.4f}")
            logger.info("-" * 50)

        # Save best model checkpoint (rank 0 only)
        if val_f1_micro > best_val_f1 and is_main_process():
            best_val_f1 = val_f1_micro
            # Gather full state dict from FSDP shards and save on rank 0
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state = model.state_dict()
            torch.save(state, f'{output_dir}/best_temporal_model_{dataset_name}.pt')

        dist.barrier()

    return model

def evaluate_model(model, test_loader, dataset_name, logger=None):
    """Evaluate the model on test set."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model.eval()
    test_preds = []
    test_labels_list = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            labels = batch['labels'].to(local_rank)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    f1_micro = f1_score(test_labels_list, test_preds, average='micro')
    f1_macro = f1_score(test_labels_list, test_preds, average='macro')
    label_names = DATASET_CONFIGS[dataset_name]['labels']
    report = classification_report(test_labels_list, test_preds, target_names=label_names, digits=4)

    if is_main_process() and logger:
        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Micro-F1: {f1_micro:.4f}")
        logger.info(f"Macro-F1: {f1_macro:.4f}")
        logger.info("Detailed Classification Report:")
        logger.info(f"\n{report}")

    return f1_micro, f1_macro, report

def main():
    # ---- Distributed setup ----
    local_rank = setup_dist()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Temporal Relation Classification - Full Fine-Tuning with FSDP')
    parser.add_argument('--dataset', type=str, default='matres', choices=list(DATASET_CONFIGS.keys()),
                      help='Dataset to use (matres, tbd, tcr, tdd_auto, tdd_man)')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                      help='Model name (e.g., roberta-base, meta-llama/Llama-3.1-8B, etc.)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Per-GPU batch size for training and evaluation')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for training')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory for results')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    set_random_seed(args.seed)

    if is_main_process():
        logger = setup_logging(args.output_dir)
        logger.info("Training Arguments:")
        logger.info("-" * 40)
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")
        logger.info("-" * 40)
        logger.info(f"World size: {dist.get_world_size()}")
        logger.info(f"Local rank: {local_rank}")
    else:
        logger = None

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and split data (all ranks read, but only rank 0 prints)
    train_data, val_data, test_data = load_and_split_data(args.dataset)
    num_labels = len(DATASET_CONFIGS[args.dataset]['labels'])

    # 3. Create datasets (single-input: event1 [SEP] event2)
    use_routing = False
    train_dataset = TemporalRelationDataset(train_data, tokenizer, args.dataset, use_routing=use_routing)
    val_dataset   = TemporalRelationDataset(val_data,   tokenizer, args.dataset, use_routing=use_routing)
    test_dataset  = TemporalRelationDataset(test_data,  tokenizer, args.dataset, use_routing=use_routing)

    collate_single = lambda batch: collate_fn_single_merged(
        batch, tokenizer=tokenizer, max_length=256, pad_to_multiple_of=8
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
    test_sampler  = DistributedSampler(test_dataset,  shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=collate_single, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, sampler=val_sampler,
                              collate_fn=collate_single, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, sampler=test_sampler,
                              collate_fn=collate_single, num_workers=2, pin_memory=True)

    # 4. Build model (bf16 weights loaded on CPU first to avoid OOM on a single GPU)
    if is_main_process() and logger:
        logger.info(f"Loading model: {args.model_name}")

    model = build_model(args.model_name, num_labels, logger=logger)

    # 5. Wrap with FSDP FULL_SHARD + bf16
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    layer_cls = get_transformer_layer_cls(model)
    if layer_cls is not None:
        auto_wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
    else:
        auto_wrap = None

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=bf16_policy,
        auto_wrap_policy=auto_wrap,
        device_id=local_rank,
        use_orig_params=True,   # keeps param names intact for optimizer
    )

    if is_main_process() and logger:
        logger.info("FSDP model ready (FULL_SHARD, bf16)")

    # 6. Train
    if is_main_process() and logger:
        logger.info("Starting training...")
    model = train_model(
        model, train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        logger=logger,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataset_name=args.dataset
    )

    # 7. Load best checkpoint (rank 0 saved it) and evaluate
    if is_main_process() and logger:
        logger.info("Loading best model for evaluation...")

    dist.barrier()
    best_ckpt = f'{args.output_dir}/best_temporal_model_{args.dataset}.pt'
    if os.path.exists(best_ckpt):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            if is_main_process():
                state = torch.load(best_ckpt, map_location='cpu')
                model.load_state_dict(state)
    dist.barrier()

    f1_micro, f1_macro, report = evaluate_model(model, test_loader, args.dataset, logger)

    # 8. Save results (rank 0 only)
    if is_main_process():
        with open(f'{args.output_dir}/temporal_relation_results.txt', 'w') as f:
            f.write(f"TEMPORAL RELATION CLASSIFICATION RESULTS - Dataset: {args.dataset}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Micro-F1: {f1_micro:.4f}\n")
            f.write(f"Macro-F1: {f1_macro:.4f}\n")
            f.write("\nDetailed Classification Report:\n")
            f.write(report)

        if logger:
            logger.info(f"Results saved to '{args.output_dir}/temporal_relation_results.txt'")
            logger.info(f"Final Test Micro-F1: {f1_micro:.4f}")
            logger.info(f"Final Test Macro-F1: {f1_macro:.4f}")

        # Clean up checkpoint
        try:
            if os.path.exists(best_ckpt):
                os.remove(best_ckpt)
                if logger:
                    logger.info(f"Cleaned up checkpoint: {best_ckpt}")
        except Exception as e:
            if logger:
                logger.warning(f"Could not delete checkpoint: {e}")

        if logger:
            logger.info("Training and evaluation completed successfully!")
            logger.info("="*60)

    cleanup_dist()

if __name__ == "__main__":
    main()  