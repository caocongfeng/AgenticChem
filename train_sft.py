#!/usr/bin/env python3
"""Train and evaluate the AgenticChem curated-vs-original COT ablation."""

# Keep GPU visibility configuration before importing torch/transformers.
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import gc
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
except ImportError:
    pass


DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
REQUIRED_FIELDS = ("explain", "smile", "cot")
PROMPT = """You are an expert in chemistry. Given the following molecular description, write its SMILES representation step-by-step. Analyze the main structure, then the substructures, then confirm other molecular properties. Your final answer must be exactly in the form <answer>{{smile}}</answer>.

Molecular description:
{explain}"""

GENERIC_CHAT_TEMPLATE = """{% for message in messages %}{{ message['role'] + ':\n' + message['content'] + eos_token + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:\n' }}{% endif %}"""
ANSWER_RE = re.compile(
    r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
            if missing:
                raise ValueError(f"Missing/empty fields {missing} in {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def paired_records(original_path: Path, curated_path: Path):
    """Align datasets so that only the COT differs between conditions."""
    original, curated = read_jsonl(original_path), read_jsonl(curated_path)

    def key(row):
        return row["explain"], row["smile"]

    original_by_key, curated_by_key = defaultdict(list), defaultdict(list)
    for row in original:
        original_by_key[key(row)].append(row)
    for row in curated:
        curated_by_key[key(row)].append(row)

    common_keys = sorted(set(original_by_key) & set(curated_by_key))
    pairs = []
    for item_key in common_keys:
        n = min(len(original_by_key[item_key]), len(curated_by_key[item_key]))
        pairs.extend(zip(original_by_key[item_key][:n], curated_by_key[item_key][:n]))
    if not pairs:
        raise ValueError("No common (explain, smile) records found between datasets")

    return pairs, {
        "original_rows": len(original),
        "curated_rows": len(curated),
        "paired_rows": len(pairs),
        "paired_unique_keys": len(common_keys),
    }


def build_messages(row: dict[str, Any], include_answer: bool = True):
    result = [{"role": "user", "content": PROMPT.format(explain=row["explain"])}]
    if include_answer:
        result.append({"role": "assistant", "content": row["cot"]})
    return result


def build_plain_prompt(row: dict[str, Any]):
    return f"User:\n{PROMPT.format(explain=row['explain'])}\n\nAssistant:\n"


def build_plain_completion(row: dict[str, Any], tokenizer):
    return row["cot"] + tokenizer.eos_token


def build_plain_text(row: dict[str, Any], tokenizer, include_answer: bool = True):
    text = build_plain_prompt(row)
    if include_answer:
        text += build_plain_completion(row, tokenizer)
    return text


def render_prompt(row: dict[str, Any], tokenizer, format_style: str):
    if format_style == "plain":
        return build_plain_text(row, tokenizer, include_answer=False)
    return tokenizer.apply_chat_template(
        build_messages(row, include_answer=False),
        tokenize=False,
        add_generation_prompt=True,
    )


def configure_tokenizer(model_name: str, format_style: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if format_style == "chat" and not tokenizer.chat_template:
        tokenizer.chat_template = GENERIC_CHAT_TEMPLATE
        print("Tokenizer had no chat template; installed a generic role-label template.")
    return tokenizer


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    candidate = matches[-1].strip() if matches else ""
    if not candidate and text:
        nonempty = [line.strip() for line in text.splitlines() if line.strip()]
        candidate = nonempty[-1] if nonempty else ""
    candidate = candidate.replace("```smiles", "").replace("```", "").strip()
    return candidate.splitlines()[0].strip() if candidate else ""


def canonicalize(smiles: str):
    from rdkit import Chem

    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule else None


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    if not smiles_a or not smiles_b:
        return 0.0
    molecule_a, molecule_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if molecule_a is None or molecule_b is None:
        return 0.0
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(molecule_a), generator.GetFingerprint(molecule_b)
        )
    )


def score_candidate(generation: str, reference_canonical: str):
    prediction = extract_answer(generation)
    prediction_canonical = canonicalize(prediction)
    valid = prediction_canonical is not None
    exact = bool(valid and prediction_canonical == reference_canonical)
    similarity = tanimoto(prediction_canonical, reference_canonical) if valid else 0.0
    return {
        "generation": generation,
        "prediction": prediction,
        "canonical_prediction": prediction_canonical,
        "valid": valid,
        "exact_match": exact,
        "similarity": similarity,
    }


def load_model(model_name: str):
    """Load a Hugging Face base model or a locally saved PEFT adapter."""
    model_path = Path(model_name)
    common = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "trust_remote_code": True,
    }

    def load_base(base_name):
        config = AutoConfig.from_pretrained(base_name, trust_remote_code=True)
        if config.model_type in {"mistral3", "gemma4"}:
            return AutoModelForMultimodalLM.from_pretrained(base_name, **common)
        return AutoModelForCausalLM.from_pretrained(base_name, **common)

    if (model_path / "adapter_config.json").exists():
        adapter_config = json.loads(
            (model_path / "adapter_config.json").read_text(encoding="utf-8")
        )
        base_model = load_base(adapter_config["base_model_name_or_path"])
        return PeftModel.from_pretrained(base_model, str(model_path))
    return load_base(model_name)


def generation_kwargs(tokenizer, max_new_tokens: int, do_sample: bool, args):
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    return kwargs


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    rows,
    output_path: Path,
    args,
    metric_prefix: str = "test",
    evaluation_epoch: int | None = None,
    global_step: int | None = None,
):
    """Evaluate greedy Best-of-1 and sampled Best-of-N molecular quality."""
    if args.best_of_n < 1 or args.best_of_n & (args.best_of_n - 1):
        raise ValueError("--best-of-n must be a positive power of two")

    reference_canonical = [canonicalize(row["smile"]) for row in rows]
    invalid_references = [i for i, value in enumerate(reference_canonical) if value is None]
    if invalid_references:
        raise ValueError(f"Invalid reference SMILES at test indices: {invalid_references[:20]}")

    n_values = []
    n = args.best_of_n
    while n >= 1:
        n_values.append(n)
        n //= 2
    sample_metrics = {n: [] for n in n_values}
    predictions = []

    model.eval()
    model.config.get_text_config().use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    try:
        model.gradient_checkpointing_disable()
    except (AttributeError, ValueError):
        pass

    for start in tqdm(
        range(0, len(rows), args.eval_batch_size), desc="Test generation"
    ):
        batch = rows[start : start + args.eval_batch_size]
        prompts = [
            render_prompt(row, tokenizer, args.format_style)
            for row in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        ).to(next(model.parameters()).device)
        prompt_length = inputs.input_ids.shape[1]

        greedy_ids = model.generate(
            **inputs,
            **generation_kwargs(tokenizer, args.max_new_tokens, False, args),
        )
        greedy_texts = tokenizer.batch_decode(
            greedy_ids[:, prompt_length:], skip_special_tokens=True
        )

        sampled_texts = [[] for _ in batch]
        if args.best_of_n > 1:
            for sample_index in range(args.best_of_n):
                # Keep sampled decoding randomness paired across curated/original.
                set_seed(args.eval_seed + start * args.best_of_n + sample_index)
                sampled_ids = model.generate(
                    **inputs,
                    **generation_kwargs(tokenizer, args.max_new_tokens, True, args),
                )
                decoded = tokenizer.batch_decode(
                    sampled_ids[:, prompt_length:], skip_special_tokens=True
                )
                for local_index, text in enumerate(decoded):
                    sampled_texts[local_index].append(text)

        for local_index, row in enumerate(batch):
            global_index = start + local_index
            ref_canonical = reference_canonical[global_index]
            greedy = score_candidate(greedy_texts[local_index], ref_canonical)
            sampled = [
                score_candidate(text, ref_canonical)
                for text in sampled_texts[local_index]
            ]
            per_n = {}
            for current_n in n_values:
                candidates = [greedy] if current_n == 1 else sampled[:current_n]
                metric = {
                    "EM": float(any(item["exact_match"] for item in candidates)),
                    "similarity": float(
                        max((item["similarity"] for item in candidates), default=0.0)
                    ),
                    "validation": float(any(item["valid"] for item in candidates)),
                }
                sample_metrics[current_n].append(metric)
                per_n[str(current_n)] = metric
            predictions.append(
                {
                    "index": global_index,
                    "explain": row["explain"],
                    "reference": row["smile"],
                    "canonical_reference": ref_canonical,
                    "metrics": per_n,
                    "greedy": greedy,
                    "sampled": sampled,
                }
            )

    aggregate = {
        str(current_n): {
            metric: float(np.mean([item[metric] for item in sample_metrics[current_n]]))
            for metric in ("EM", "similarity", "validation")
        }
        for current_n in n_values
    }
    result = {"n": len(rows), "best_of": aggregate}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"metrics": result, "predictions": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if wandb.run is not None:
        payload = {}
        if evaluation_epoch is not None:
            payload["test/epoch"] = evaluation_epoch
        if global_step is not None:
            payload["test/global_step"] = global_step
        for current_n, metrics in aggregate.items():
            for metric, value in metrics.items():
                payload[f"{metric_prefix}/best_of_{current_n}/{metric}"] = value
        wandb.log(payload)
        for key, value in payload.items():
            wandb.run.summary[key] = value
    print(json.dumps(result, indent=2))
    return result


class EpochTestCallback(TrainerCallback):
    """Run the complete generation-based test evaluation after every epoch."""

    def __init__(
        self,
        tokenizer,
        rows,
        output_dir: Path,
        eval_args,
        eval_every_epoch: bool = True,
    ):
        self.tokenizer = tokenizer
        self.rows = rows
        self.output_dir = output_dir
        self.eval_args = eval_args
        self.eval_every_epoch = eval_every_epoch
        self.latest_prediction_path = None
        self.latest_result = None
        self.prediction_paths_by_step = {}
        self.results_by_step = {}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.epoch is None:
            return control
        if (
            not self.eval_every_epoch
            and state.epoch < float(args.num_train_epochs) - 1e-6
        ):
            return control

        rounded_epoch = round(state.epoch)
        epoch_number = int(
            rounded_epoch
            if math.isclose(state.epoch, rounded_epoch, abs_tol=1e-6)
            else math.ceil(state.epoch)
        )
        output_path = self.output_dir / f"test_predictions_epoch_{epoch_number:03d}.json"
        print(f"Running complete test evaluation after epoch {state.epoch:g}...")

        # Full generation changes RNG/model state; restore it before training continues.
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        was_training = model.training
        use_cache = model.config.get_text_config().use_cache
        was_gradient_checkpointing = getattr(model, "is_gradient_checkpointing", False)
        generation_config = getattr(model, "generation_config", None)
        generation_max_length = (
            generation_config.max_length if generation_config is not None else None
        )
        try:
            set_seed(self.eval_args.eval_seed)
            result = evaluate(
                model,
                self.tokenizer,
                self.rows,
                output_path,
                self.eval_args,
                metric_prefix="test",
                evaluation_epoch=epoch_number,
                global_step=int(state.global_step),
            )
            self.latest_prediction_path = output_path
            self.latest_result = result
            self.prediction_paths_by_step[state.global_step] = output_path
            self.results_by_step[state.global_step] = result
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            model.config.get_text_config().use_cache = use_cache
            if generation_config is not None:
                generation_config.max_length = generation_max_length
            if was_gradient_checkpointing:
                try:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    model.gradient_checkpointing_enable()
            model.train(was_training)

        return control


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--experiment-name")
    parser.add_argument("--format-style", choices=("plain", "chat"), default="plain")
    parser.add_argument("--data-dir", default="AgenticChem")
    parser.add_argument("--variant", choices=("curated", "original"), required=True)
    parser.add_argument("--output-dir", default="AgenticChem/runs")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--n-runs", type=int)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--best-of-n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--limit-train", type=int, default=-1)
    parser.add_argument("--limit-val", type=int, default=-1)
    parser.add_argument("--limit-test", type=int, default=-1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-before-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run test generation on the untrained model before SFT.",
    )
    parser.add_argument(
        "--eval-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run complete test generation after every epoch. With "
            "--no-eval-every-epoch, run it only after the last epoch."
        ),
    )
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--allow-unpaired", action="store_true")
    parser.add_argument("--allow-split-overlap", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "AgenticChem-COT"))
    parser.add_argument("--wandb-entity", default=os.getenv("c-cao-university-of-amsterdam"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.grad_accum < 1:
        raise ValueError("Batch sizes and gradient accumulation must be >= 1")
    seed = args.seed if args.seed is not None else 1000 + args.run_id
    eval_seed = args.eval_seed if args.eval_seed is not None else 2000 + args.run_id
    args.eval_seed = eval_seed
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / (
        "curated_train_data.jsonl" if args.variant == "curated" else "train_data.jsonl"
    )
    val_path = data_dir / (
        "curated_val_data.jsonl" if args.variant == "curated" else "val_data.jsonl"
    )
    test_path = data_dir / "test_data.jsonl"

    test_rows = read_jsonl(test_path)
    if args.allow_unpaired:
        train_rows, val_rows = read_jsonl(train_path), read_jsonl(val_path)
        train_stats = {"pairing": "disabled", "selected_rows": len(train_rows)}
        val_stats = {"pairing": "disabled", "selected_rows": len(val_rows)}
    else:
        train_pairs, train_stats = paired_records(
            data_dir / "train_data.jsonl", data_dir / "curated_train_data.jsonl"
        )
        val_pairs, val_stats = paired_records(
            data_dir / "val_data.jsonl", data_dir / "curated_val_data.jsonl"
        )
        pair_index = 0 if args.variant == "original" else 1
        train_rows = [pair[pair_index] for pair in train_pairs]
        val_rows = [pair[pair_index] for pair in val_pairs]

    if not args.allow_split_overlap:
        row_key = lambda row: (row["explain"], row["smile"])
        test_keys = {row_key(row) for row in test_rows}
        train_before, val_before = len(train_rows), len(val_rows)
        train_rows = [row for row in train_rows if row_key(row) not in test_keys]
        val_rows = [row for row in val_rows if row_key(row) not in test_keys]
        val_keys = {row_key(row) for row in val_rows}
        train_after_test = len(train_rows)
        train_rows = [row for row in train_rows if row_key(row) not in val_keys]
        train_stats.update(
            {
                "removed_test_overlap_rows": train_before - train_after_test,
                "removed_val_overlap_rows": train_after_test - len(train_rows),
            }
        )
        val_stats["removed_test_overlap_rows"] = val_before - len(val_rows)

    if args.limit_train > 0:
        train_rows = train_rows[: args.limit_train]
    if args.limit_val > 0:
        val_rows = val_rows[: args.limit_val]
    if args.limit_test > 0:
        test_rows = test_rows[: args.limit_test]

    experiment_name = args.experiment_name or args.model.replace("/", "__")
    output_dir = (
        Path(args.output_dir)
        / experiment_name
        / args.variant
        / f"run_{args.run_id:03d}"
    )
    model_dir = output_dir / "model"
    if (
        not args.eval_only
        and not args.resume_from_checkpoint
        and model_dir.exists()
        and any(model_dir.iterdir())
    ):
        raise FileExistsError(
            f"Non-empty model directory already exists: {model_dir}. "
            "Use a new run ID or --resume-from-checkpoint."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "seed": seed,
        "experiment_name": experiment_name,
        "strict_paired": not args.allow_unpaired,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "test_file": str(test_path),
        "train_rows_used": len(train_rows),
        "val_rows_used": len(val_rows),
        "test_rows_used": len(test_rows),
        "train_pairing": train_stats,
        "val_pairing": val_stats,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity="c-cao-university-of-amsterdam",
            group=args.wandb_group or os.getenv("SLURM_ARRAY_JOB_ID"),
            name=args.wandb_run_name
            or f"{experiment_name}-{args.variant}-run-{args.run_id:03d}",
            config=run_config,
            tags=[
                experiment_name,
                args.variant,
                "paired" if not args.allow_unpaired else "unpaired",
            ],
        )
        wandb.run.define_metric("test/epoch")
        wandb.run.define_metric("test/*", step_metric="test/epoch")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = configure_tokenizer(args.model, args.format_style)
    model = load_model(args.model)
    baseline_prediction_path = None
    if args.eval_before_train and not args.eval_only:
        baseline_prediction_path = output_dir / "test_predictions_epoch_000.json"
        if baseline_prediction_path.exists():
            print(
                "Pre-training test result already exists; "
                f"skipping: {baseline_prediction_path}"
            )
        else:
            print("Running pre-training test evaluation (epoch 0)...")
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_states = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            was_training = model.training
            use_cache = model.config.get_text_config().use_cache
            generation_config = getattr(model, "generation_config", None)
            generation_max_length = (
                generation_config.max_length
                if generation_config is not None
                else None
            )
            try:
                set_seed(eval_seed)
                evaluate(
                    model,
                    tokenizer,
                    test_rows,
                    baseline_prediction_path,
                    args,
                    metric_prefix="test",
                    evaluation_epoch=0,
                    global_step=0,
                )
            finally:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.random.set_rng_state(torch_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                model.config.get_text_config().use_cache = use_cache
                if generation_config is not None:
                    generation_config.max_length = generation_max_length
                model.train(was_training)
    epoch_test_callback = None
    selected_prediction_path = None
    selected_test_result = None
    if not args.eval_only:
        if args.format_style == "plain":
            train_dataset = Dataset.from_dict(
                {
                    "prompt": [build_plain_prompt(row) for row in train_rows],
                    "completion": [
                        build_plain_completion(row, tokenizer) for row in train_rows
                    ],
                }
            )
            val_dataset = Dataset.from_dict(
                {
                    "prompt": [build_plain_prompt(row) for row in val_rows],
                    "completion": [
                        build_plain_completion(row, tokenizer) for row in val_rows
                    ],
                }
            )
        else:
            train_dataset = Dataset.from_list(train_rows).map(
                lambda row: {"messages": build_messages(row)}
            )
            val_dataset = Dataset.from_list(val_rows).map(
                lambda row: {"messages": build_messages(row)}
            )
        peft_config = None
        if not args.full_finetune:
            peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=2 * args.lora_rank,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules="all-linear",
                exclude_modules=r".*(vision_tower|vision_encoder|multi_modal_projector|multimodal_projector|audio_tower).*",
                task_type="CAUSAL_LM",
            )
        updates_per_epoch = math.ceil(
            len(train_dataset) / (args.batch_size * args.grad_accum)
        )
        total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
        warmup_steps = (
            max(1, int(total_updates * args.warmup_ratio))
            if args.warmup_ratio > 0
            else 0
        )
        training_args = SFTConfig(
            output_dir=str(model_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            optim="adamw_torch_fused",
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=args.max_seq_length,
            completion_only_loss=args.format_style == "plain",
            packing=False,
            report_to=[] if args.no_wandb else ["wandb"],
            seed=seed,
            data_seed=seed,
        )
        model.config.get_text_config().use_cache = False
        epoch_test_callback = EpochTestCallback(
            tokenizer,
            test_rows,
            output_dir,
            args,
            eval_every_epoch=args.eval_every_epoch,
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            callbacks=[epoch_test_callback],
        )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        selected_prediction_path = epoch_test_callback.prediction_paths_by_step.get(
            trainer.state.best_global_step,
            epoch_test_callback.latest_prediction_path,
        )
        selected_test_result = epoch_test_callback.results_by_step.get(
            trainer.state.best_global_step,
            epoch_test_callback.latest_result,
        )
        # SFTTrainer wraps the base model with PEFT internally. Keep that wrapped,
        # trained instance for both saving and evaluation.
        model = trainer.model
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        trainer.optimizer = None
        trainer.lr_scheduler = None
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_path = output_dir / "test_predictions.json"
    if selected_prediction_path is not None:
        shutil.copyfile(selected_prediction_path, prediction_path)
        if wandb.run is not None and selected_test_result is not None:
            for current_n, metrics in selected_test_result["best_of"].items():
                for metric, value in metrics.items():
                    wandb.run.summary[f"test/best_of_{current_n}/{metric}"] = value
    else:
        set_seed(eval_seed)
        evaluate(model, tokenizer, test_rows, prediction_path, args)

    runtime_metrics = {}
    if torch.cuda.is_available():
        runtime_metrics = {
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        print(json.dumps(runtime_metrics, indent=2))
        if wandb.run is not None:
            wandb.log({f"system/{key}": value for key, value in runtime_metrics.items()})
            for key, value in runtime_metrics.items():
                wandb.run.summary[f"system/{key}"] = value
    runtime_path = output_dir / "runtime_metrics.json"
    runtime_path.write_text(json.dumps(runtime_metrics, indent=2), encoding="utf-8")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            f"{experiment_name}-{args.variant}-run-{args.run_id:03d}-results",
            type="evaluation",
        )
        artifact.add_file(str(prediction_path))
        if (
            baseline_prediction_path is not None
            and baseline_prediction_path.exists()
        ):
            artifact.add_file(str(baseline_prediction_path))
        artifact.add_file(str(config_path))
        artifact.add_file(str(runtime_path))
        wandb.log_artifact(artifact, aliases=["latest"])
        wandb.finish()


if __name__ == "__main__":
    main()
