"""Private Kaggle SFT kernel. GGUF conversion runs separately on the server."""

import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

MODEL = "openbmb/MiniCPM5-2B"
UNSLOTH_VERSION = "2026.9.12"
MAX_SEQ_LENGTH = 4096
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def render_sample(sample, tokenizer, max_seq_length=MAX_SEQ_LENGTH):
    """Render runtime messages and label each assistant JSON and turn terminator."""
    messages = sample["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) > max_seq_length:
        return None
    spans, answers = [], []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        answer = message["content"]
        value = json.loads(answer)
        if not isinstance(value, dict) or set(value) != {"thought", "tool", "args"}:
            raise ValueError("Assistant content must be a JSON step")
        prefix = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if not prefix.endswith(RESPONSE_PART) or not text.startswith(prefix + answer):
            raise ValueError("Chat template differs from the runtime assistant prefix")
        end = len(prefix) + len(answer)
        terminator = text[end:end + len("<|im_end|>")]
        if terminator != "<|im_end|>":
            raise ValueError("Chat template differs from the assistant turn terminator")
        spans.append((len(prefix), end + len(terminator)))
        answers.append(answer + terminator)
    labels = [
        token if any(start <= left < right <= end for start, end in spans) else -100
        for token, (left, right) in zip(
            encoded["input_ids"], encoded["offset_mapping"], strict=True
        )
    ]
    decoded = tokenizer.decode(
        [token for token in labels if token != -100],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not answers or decoded != "".join(answers):
        raise ValueError("Token boundaries do not preserve the assistant JSON and terminators")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def step_control(state, control, deadline):
    """Finish an update, then evaluate/save before the trainer reloads its best model."""
    if time.monotonic() >= deadline:
        control.should_training_stop = True
    if control.should_training_stop or state.global_step >= state.max_steps:
        control.should_save = control.should_evaluate = control.should_log = True
    return control


def main():
    started = time.monotonic()
    deadline = started + 10 * 3600
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"unsloth=={UNSLOTH_VERSION}",
         "unsloth_zoo==2026.9.8", "trl==0.24.0", "datasets==4.3.0", "transformers==4.57.6"],
        check=True,
    )
    # Unsloth must patch Transformers before TRL or Transformers is imported.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    # isort: split
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class TimeGuard(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            return step_control(state, control, deadline)

        def on_evaluate(self, args, state, control, **kwargs):
            if time.monotonic() >= deadline:
                control.should_training_stop = True
            return control

    WORK.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_SEQ_LENGTH, dtype=torch.float16,
        load_in_4bit=True, use_exact_model_name=True, fix_tokenizer=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    datasets, counts, smoke_samples = {}, {}, []
    sources = [path for path in INPUT.rglob("sft_train.jsonl")
               if path.with_name("sft_val.jsonl").is_file()]
    if len(sources) != 1:
        raise ValueError("Expected one input directory containing both SFT splits")
    for split in ("train", "val"):
        rows, total = [], 0
        with sources[0].with_name(f"sft_{split}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                sample = json.loads(line)
                total += 1
                row = render_sample(sample, tokenizer)
                if row is not None:
                    rows.append(row)
                    if split == "val" and len(smoke_samples) < 5:
                        smoke_samples.append(sample)
        counts[split] = {"used": len(rows), "dropped": total - len(rows)}
        print(f"{split}: used {len(rows)}, dropped {total - len(rows)} over 4096 tokens")
        if not rows:
            raise ValueError(f"No {split} samples remain after length filtering")
        datasets[split] = Dataset.from_dict({key: [row[key] for row in rows] for key in rows[0]})
    if len(datasets["train"]) < 3 or len(smoke_samples) < 5:
        raise ValueError("Need at least 3 training and 5 validation samples for the checks")
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["val"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
        callbacks=[TimeGuard()],
        args=SFTConfig(
            output_dir=str(WORK / "checkpoints"), max_length=MAX_SEQ_LENGTH,
            dataset_kwargs={"skip_prepare_dataset": True}, packing=False,
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_ratio=0.03, fp16=True, bf16=False, gradient_checkpointing=True,
            per_device_eval_batch_size=1, fp16_full_eval=True, prediction_loss_only=True,
            eval_strategy="steps", eval_steps=200, save_strategy="steps", save_steps=200,
            save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            logging_steps=10, seed=42, report_to="none",
        ),
    )
    # Existing labels train turn terminators, masking newlines and user marker-like text.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
        force_match=True, num_proc=1,
    )
    for split, masked in (("train", trainer.train_dataset), ("val", trainer.eval_dataset)):
        for original, row in zip(datasets[split], masked, strict=True):
            if row["labels"] != original["labels"]:
                raise ValueError(
                    f"Unsloth masking changed the JSON and terminator labels in {split}"
                )
    for i in range(3):
        print(f"MASKED {i}: " + tokenizer.decode(
            [token for token in trainer.train_dataset[i]["labels"] if token != -100],
            skip_special_tokens=False, clean_up_tokenization_spaces=False,
        ))
    trained = False
    try:
        if time.monotonic() < deadline:
            trainer.train()
            trained = True
        else:
            print("Time guard expired during setup; saving the initialized adapter")
    finally:
        summary = {
            "model": MODEL, "samples": counts, "steps": trainer.state.global_step,
            "best_eval_loss": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "training_returned": trained,
            "wall_seconds": round(time.monotonic() - started, 2),
            "time_limit_reached": time.monotonic() >= deadline,
            "versions": {name: version(name) for name in (
                "unsloth", "unsloth_zoo", "trl", "datasets", "transformers", "torch",
                "peft", "bitsandbytes", "accelerate", "safetensors",
            )},
        }
        (WORK / "trainer_logs.json").write_text(
            json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
        )
        (WORK / "train_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        model.save_pretrained(str(WORK / "adapter"), safe_serialization=True)
        tokenizer.save_pretrained(str(WORK / "adapter"))
        # None forces safetensors even on Unsloth's low-CPU pickle fallback.
        model.save_pretrained_merged(
            str(WORK / "merged"), tokenizer, save_method="merged_16bit",
            safe_serialization=None, maximum_memory_usage=0.5,
        )
        tokenizer.save_pretrained(str(WORK / "merged"))
    FastLanguageModel.for_inference(model)
    eos_token_ids = model.generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    for i, sample in enumerate(smoke_samples):
        first = next(j for j, message in enumerate(sample["messages"])
                     if message["role"] == "assistant")
        inputs = tokenizer.apply_chat_template(
            sample["messages"][:first], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = output[0, inputs["input_ids"].shape[1]:]
        ended_on_eos = int(generated[-1]) in eos_token_ids
        hit_max_new_tokens = len(generated) == 512 and not ended_on_eos
        text = tokenizer.decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        try:
            value = json.loads(text)
            valid = isinstance(value, dict) and set(value) == {"thought", "tool", "args"}
        except ValueError:
            valid = False
        print(f"SMOKE {i}: json_step={valid} ended_on_eos={ended_on_eos} "
              f"hit_max_new_tokens={hit_max_new_tokens}\n{text}")
    summary["wall_seconds"] = round(time.monotonic() - started, 2)
    (WORK / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
