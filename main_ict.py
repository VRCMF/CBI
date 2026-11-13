import os
os.environ['CURL_CA_BUNDLE'] = ''

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, f1_score
import warnings
import argparse
import logging
import datetime
import sys
import time
import random

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
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

LABEL_DESCRIPTIONS = {
    'BEFORE':      'Event 1 occurs BEFORE Event 2',
    'AFTER':       'Event 1 occurs AFTER Event 2',
    'SIMULTANEOUS':'Event 1 and Event 2 occur at the SAME TIME',
    'INCLUDES':    'The time span of Event 1 INCLUDES the time span of Event 2',
    'IS_INCLUDED': 'The time span of Event 1 IS INCLUDED within the time span of Event 2',
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_random_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str):
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'inference_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TEMPORAL RELATION EXTRACTION - IN-CONTEXT LEARNING")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    return logger


def load_and_split_data(dataset_name: str):
    data_path = DATASET_CONFIGS[dataset_name]['path']
    df = pd.read_csv(data_path)
    valid_relations = DATASET_CONFIGS[dataset_name]['labels']
    df = df[df['relation'].isin(valid_relations)].reset_index(drop=True)

    train_data = df[df['split'] == 'train'].reset_index(drop=True)
    val_data   = df[df['split'] == 'validation'].reset_index(drop=True)
    test_data  = df[df['split'] == 'test'].reset_index(drop=True)
    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------
def select_few_shot_examples(train_data: pd.DataFrame, num_shots: int,
                              labels: list, seed: int = 42) -> list:
    """
    Sample `num_shots` examples balanced across label classes.
    Returns a list of dicts with keys: text1, text2, label.
    """
    if num_shots == 0:
        return []

    shots_per_class = max(1, num_shots // len(labels))
    remainder = num_shots - shots_per_class * len(labels)

    examples = []
    for i, label in enumerate(labels):
        label_data = train_data[train_data['relation'] == label]
        n = shots_per_class + (1 if i < remainder else 0)
        n = min(n, len(label_data))
        if n > 0:
            sampled = label_data.sample(n=n, random_state=seed)
            for _, row in sampled.iterrows():
                examples.append({
                    'text1': str(row['event_ctx_1']).strip(),
                    'text2': str(row['event_ctx_2']).strip(),
                    'label': str(row['relation']).strip()
                })

    random.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_system_instruction(labels: list) -> str:
    label_lines = '\n'.join(
        f"  - {lbl}: {LABEL_DESCRIPTIONS.get(lbl, lbl)}" for lbl in labels
    )
    valid_str = ', '.join(labels)
    return (
        "You are an expert in temporal information extraction. "
        "Determine the temporal relation between the two events based on their contexts.\n\n"
        f"Possible relations:\n{label_lines}\n\n"
        f"Reply with ONLY one label from: {valid_str}. No explanation."
    )


def _format_qa(text1: str, text2: str, label: str = None) -> str:
    q = f"Event 1 context: {text1}\nEvent 2 context: {text2}\nTemporal relation:"
    return q + (f" {label}" if label else "")


def _apply_chat_template_with_fallback(tokenizer, messages: list,
                                       system_instr: str) -> str:
    """
    Try apply_chat_template with a system message.
    If the model's template raises 'System role not supported'
    (e.g. Gemma), fold the system instruction into the first user message
    and retry without the system role.
    """
    import jinja2
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (jinja2.exceptions.TemplateError, ValueError) as e:
        if "system" not in str(e).lower() and "role" not in str(e).lower():
            raise
        # Model does not support system role — merge system instr into first user turn
        merged = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # drop system message; will be prepended below
            if msg["role"] == "user" and not merged:
                # Prepend system instruction to the very first user message
                merged.append({"role": "user",
                                "content": system_instr + "\n\n" + msg["content"]})
            else:
                merged.append(msg)
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True
        )


def build_prompt(text1: str, text2: str, labels: list,
                 few_shot_examples: list = None,
                 use_chat_template: bool = False,
                 tokenizer=None) -> str:
    """
    Build an inference prompt for zero-shot or few-shot ICL.

    If `use_chat_template` is True and the tokenizer has a chat template,
    the prompt is formatted as a multi-turn chat.  Models that don't support
    a system role (e.g. Gemma) have the system instruction folded into the
    first user turn automatically.
    """
    system_instr = _build_system_instruction(labels)
    few_shot_examples = few_shot_examples or []

    if use_chat_template and tokenizer is not None and \
            getattr(tokenizer, 'chat_template', None) is not None:
        messages = [{"role": "system", "content": system_instr}]
        for ex in few_shot_examples:
            messages.append({"role": "user",      "content": _format_qa(ex['text1'], ex['text2'])})
            messages.append({"role": "assistant", "content": ex['label']})
        messages.append({"role": "user", "content": _format_qa(text1, text2)})

        return _apply_chat_template_with_fallback(tokenizer, messages, system_instr)

    # ---- Plain completion format ----
    parts = [system_instr + "\n\n"]
    if few_shot_examples:
        parts.append("Examples:\n")
        for i, ex in enumerate(few_shot_examples, 1):
            parts.append(f"Example {i}:\n{_format_qa(ex['text1'], ex['text2'], ex['label'])}\n\n")
    parts.append(_format_qa(text1, text2))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------
def extract_label(generated_text: str, labels: list) -> str | None:
    """
    Parse the model's raw output and return the matching label, or None.
    Sorts labels by descending length first to avoid prefix ambiguity
    (e.g. IS_INCLUDED vs INCLUDED).
    """
    text_upper = generated_text.strip().upper()
    sorted_labels = sorted(labels, key=len, reverse=True)

    # Prefer a label at the very start of the generated text
    for lbl in sorted_labels:
        if text_upper.startswith(lbl):
            return lbl

    # Fall back to any occurrence in the text
    for lbl in sorted_labels:
        if lbl in text_upper:
            return lbl

    return None


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------
def _get_input_device(model) -> torch.device:
    """Return the device where inputs should be placed (first parameter's device)."""
    # Unwrap DataParallel to reach the actual module
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    try:
        return next(base.parameters()).device
    except StopIteration:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _unwrap(model):
    """Return the underlying model, stripping DataParallel if present."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def run_inference_batch(model, tokenizer, prompts: list,
                        max_new_tokens: int = 20) -> list:
    """Tokenise `prompts`, generate, and decode only the new tokens."""
    input_device = _get_input_device(model)
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=2048
    ).to(input_device)

    prompt_len = inputs['input_ids'].shape[1]

    # Use the unwrapped model for generate() — DataParallel does not expose it
    gen_model = _unwrap(model)
    with torch.inference_mode():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_ict(model, tokenizer,
                 test_data: pd.DataFrame,
                 dataset_name: str,
                 few_shot_examples: list,
                 batch_size: int = 8,
                 max_new_tokens: int = 20,
                 use_chat_template: bool = False,
                 logger=None):
    """Run ICL inference over the entire test set and compute metrics."""
    labels    = DATASET_CONFIGS[dataset_name]['labels']
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}

    all_preds, all_true = [], []
    failed_count = 0
    n_samples = len(test_data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    start_time = time.time()

    for b_idx in range(n_batches):
        b_start = b_idx * batch_size
        b_end   = min(b_start + batch_size, n_samples)
        batch   = test_data.iloc[b_start:b_end]

        prompts, true_labels = [], []
        for _, row in batch.iterrows():
            text1      = str(row['event_ctx_1']).strip()
            text2      = str(row['event_ctx_2']).strip()
            true_label = str(row['relation']).strip()
            prompt = build_prompt(
                text1, text2, labels,
                few_shot_examples=few_shot_examples,
                use_chat_template=use_chat_template,
                tokenizer=tokenizer
            )
            prompts.append(prompt)
            true_labels.append(true_label)

        # Run batch inference; fall back to single-sample on OOM / error
        try:
            generated = run_inference_batch(model, tokenizer, prompts, max_new_tokens)
        except Exception as e:
            if logger:
                logger.warning(f"Batch {b_idx+1} failed ({e}), retrying one-by-one…")
            generated = []
            for p in prompts:
                try:
                    gen = run_inference_batch(model, tokenizer, [p], max_new_tokens)
                    generated.extend(gen)
                except Exception:
                    generated.append("")

        for gen_text, true_label in zip(generated, true_labels):
            pred_label = extract_label(gen_text, labels)
            if pred_label is None:
                failed_count += 1
                pred_label = labels[0]  # fallback to first label
            all_preds.append(label_map[pred_label])
            all_true.append(label_map[true_label])

        if logger and (b_idx + 1) % 10 == 0:
            elapsed  = time.time() - start_time
            progress = b_end / n_samples
            eta      = elapsed / progress * (1 - progress) if progress > 0 else 0
            logger.info(
                f"  [{b_end}/{n_samples} ({progress*100:.1f}%)] "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

    f1_micro = f1_score(all_true, all_preds, average='micro')
    f1_macro = f1_score(all_true, all_preds, average='macro')
    report   = classification_report(all_true, all_preds,
                                     target_names=labels, digits=4)

    if logger:
        logger.info(f"Failed/unparsed predictions: {failed_count}/{n_samples}")
        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Micro-F1: {f1_micro:.4f}")
        logger.info(f"Macro-F1: {f1_macro:.4f}")
        logger.info("Detailed Classification Report:")
        logger.info(f"\n{report}")

    return f1_micro, f1_macro, report, failed_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Temporal Relation Extraction - In-Context Learning (Zero/Few-shot)'
    )
    parser.add_argument('--dataset', type=str, default='matres',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to evaluate on')
    parser.add_argument('--model_name', type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct',
                        help='HuggingFace model name or local path')
    parser.add_argument('--num_shots', type=int, default=0,
                        help='Number of ICL examples per inference call (0 = zero-shot)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of prompts per forward pass')
    parser.add_argument('--max_new_tokens', type=int, default=20,
                        help='Maximum number of tokens to generate per sample')
    parser.add_argument('--use_chat_template', action='store_true',
                        help='Force chat-template formatting (auto-detected when not set)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to save results and logs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_test_samples', type=int, default=-1,
                        help='Cap on test samples for quick runs (-1 = all)')
    args = parser.parse_args()

    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ---- Device info ----
    n_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {n_gpus}")

    # ---- Tokenizer ----
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # required for left-padded batch generation

    # ---- Model ----
    logger.info(f"Loading model: {args.model_name}")

    # Check whether `accelerate` is available (required for device_map='auto').
    # If not, fall back to manual placement: DataParallel on all GPUs or single GPU/CPU.
    try:
        import accelerate  # noqa: F401
        has_accelerate = True
    except ImportError:
        has_accelerate = False

    if has_accelerate and n_gpus > 0:
        logger.info("Using device_map='auto' (accelerate detected, multi-GPU).")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
    elif n_gpus > 0:
        # No accelerate — load onto GPU 0, then wrap with DataParallel for multi-GPU
        logger.info(
            f"accelerate not found. Loading model on cuda:0"
            + (f" then wrapping with DataParallel across {n_gpus} GPUs." if n_gpus > 1 else ".")
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        ).to('cuda:0')
        if n_gpus > 1:
            model = torch.nn.DataParallel(model)
    else:
        logger.info("No GPU found. Loading model on CPU (will be slow).")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
        )

    model.eval()
    logger.info("Model loaded.")

    # ---- Data ----
    logger.info(f"Loading dataset: {args.dataset}")
    train_data, val_data, test_data = load_and_split_data(args.dataset)
    logger.info(f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")
    logger.info("Relation distribution (test):")
    logger.info(str(test_data['relation'].value_counts()))

    if args.max_test_samples > 0:
        test_data = test_data.sample(
            n=min(args.max_test_samples, len(test_data)),
            random_state=args.seed
        ).reset_index(drop=True)
        logger.info(f"Test set limited to {len(test_data)} samples")

    # ---- Few-shot examples ----
    labels = DATASET_CONFIGS[args.dataset]['labels']
    few_shot_examples = select_few_shot_examples(
        train_data, args.num_shots, labels, seed=args.seed
    )
    shot_type = "zero-shot" if args.num_shots == 0 else f"{args.num_shots}-shot"
    logger.info(f"Mode: {shot_type}  |  {len(few_shot_examples)} ICL examples selected")
    for i, ex in enumerate(few_shot_examples, 1):
        logger.info(f"  [{i}] label={ex['label']}  text1={ex['text1'][:60]}…")

    # ---- Chat-template auto-detection ----
    use_chat_template = args.use_chat_template
    if not use_chat_template and getattr(tokenizer, 'chat_template', None) is not None:
        logger.info("Chat template detected on tokenizer — enabling automatically.")
        use_chat_template = True

    # ---- Inference ----
    logger.info(f"Starting {shot_type} inference on {len(test_data)} samples…")
    f1_micro, f1_macro, report, failed = evaluate_ict(
        model, tokenizer,
        test_data     = test_data,
        dataset_name  = args.dataset,
        few_shot_examples = few_shot_examples,
        batch_size    = args.batch_size,
        max_new_tokens= args.max_new_tokens,
        use_chat_template = use_chat_template,
        logger        = logger,
    )

    # ---- Save results ----
    result_file = os.path.join(args.output_dir, 'ict_results.txt')
    with open(result_file, 'w') as f:
        f.write("TEMPORAL RELATION EXTRACTION - IN-CONTEXT LEARNING\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset:          {args.dataset}\n")
        f.write(f"Model:            {args.model_name}\n")
        f.write(f"Mode:             {shot_type}\n")
        f.write(f"Num shots:        {args.num_shots}\n")
        f.write(f"Use chat template:{use_chat_template}\n")
        f.write(f"Test samples:     {len(test_data)}\n")
        f.write(f"Failed preds:     {failed}\n")
        f.write(f"Micro-F1:         {f1_micro:.4f}\n")
        f.write(f"Macro-F1:         {f1_macro:.4f}\n")
        f.write("\nDetailed Classification Report:\n")
        f.write(report)

    logger.info(f"Results saved to {result_file}")
    logger.info(f"Final  Micro-F1={f1_micro:.4f}  Macro-F1={f1_macro:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
