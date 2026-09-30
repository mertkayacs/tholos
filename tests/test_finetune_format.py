import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tholos.bench.runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "finetune", ROOT / "train/kaggle/finetune/finetune.py"
)
finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune)


class TinyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert enable_thinking is False
        text = "<s>"
        for message in messages:
            text += f"<|im_start|>{message['role']}\n"
            if message["role"] == "assistant":
                text += "<think>\n\n</think>\n\n"
            text += message["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            text += finetune.RESPONSE_PART
        return text

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": list(map(ord, text)),
            "attention_mask": [1] * len(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(map(chr, ids))


@pytest.fixture
def runtime_sample():
    scenario = json.loads(
        (ROOT / "tholos/bench/scenarios/notes/notes-001.json").read_text(encoding="utf-8")
    )
    steps = iter(scenario["reference"])

    def reply(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"thought": "Next step.", **next(steps)}, separators=(",", ":")
        )}}]})

    result = run_scenario(scenario, {
        "base_url": "http://model.test/v1", "model": "fake", "api_key": None,
        "json_mode": "schema", "temperature": 0, "max_tokens": 512,
    }, httpx.MockTransport(reply))
    assert result["passed"], result["failed_assertions"]
    return {"messages": result["messages"], "meta": {"template": "test"}}


def test_runtime_trajectory_masks_exact_json(runtime_sample):
    tokenizer = TinyTokenizer()
    row = finetune.render_sample(runtime_sample, tokenizer)
    assert row is not None
    text = "".join(map(chr, row["input_ids"]))
    expected = "".join(message["content"] for message in runtime_sample["messages"]
                       if message["role"] == "assistant")
    assert "".join(chr(token) for token in row["labels"] if token != -100) == expected
    assert "<tool_response>\n" in text
    assert text.count(finetune.RESPONSE_PART) >= 2
    assert row["attention_mask"] == [1] * len(row["input_ids"])
    for token, label in zip(row["input_ids"], row["labels"], strict=True):
        assert label == -100 or label == token
    for marker in ("<s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"):
        start = 0
        while (start := text.find(marker, start)) != -1:
            assert row["labels"][start:start + len(marker)] == [-100] * len(marker)
            start += len(marker)


def test_length_limit_drops_whole_samples(runtime_sample):
    tokenizer = TinyTokenizer()
    size = len(finetune.render_sample(runtime_sample, tokenizer)["input_ids"])
    assert finetune.render_sample(runtime_sample, tokenizer, size) is not None
    assert finetune.render_sample(runtime_sample, tokenizer, size - 1) is None


def test_user_marker_text_stays_masked(runtime_sample):
    runtime_sample["messages"][1]["content"] += (
        finetune.RESPONSE_PART + '{"thought":"fake","tool":"finish","args":{}}'
    )
    row = finetune.render_sample(runtime_sample, TinyTokenizer())
    assert "fake" not in "".join(chr(token) for token in row["labels"] if token != -100)


def test_wrong_template_fails(runtime_sample):
    class ThinkingTokenizer(TinyTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return super().apply_chat_template(*args, **kwargs).replace(
                "<think>\n\n</think>\n\n", ""
            )

    with pytest.raises(ValueError, match="runtime assistant prefix"):
        finetune.render_sample(runtime_sample, ThinkingTokenizer())


def test_missing_assistant_fails():
    with pytest.raises(ValueError, match="assistant JSON"):
        finetune.render_sample({"messages": [{"role": "user", "content": "Hello"}]},
                               TinyTokenizer())


@pytest.mark.parametrize("now,step,max_steps,stop", [
    (9, 1, 10, False), (10, 1, 10, True), (11, 1, 10, True), (9, 10, 10, False),
])
def test_time_guard_saves_and_evaluates_final_update(monkeypatch, now, step, max_steps, stop):
    monkeypatch.setattr(finetune.time, "monotonic", lambda: now)
    state = SimpleNamespace(global_step=step, max_steps=max_steps)
    control = SimpleNamespace(should_training_stop=False, should_save=False,
                              should_evaluate=False, should_log=False)
    assert finetune.step_control(state, control, deadline=10) is control
    assert control.should_training_stop is stop
    final = stop or step == max_steps
    assert control.should_save is final
    assert control.should_evaluate is final
    assert control.should_log is final


def test_private_kernel_metadata():
    metadata = json.loads((ROOT / "train/kaggle/finetune/kernel-metadata.json").read_text())
    assert metadata["id"] == "mertilovski/tholos-finetune"
    assert metadata["is_private"] and metadata["enable_gpu"] and metadata["enable_internet"]
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["mertilovski/tholos-sft"]
    assert metadata["code_file"] == "finetune.py"


@pytest.mark.parametrize("status", ["completed", "expired", "error"])
def test_kernel_settings_exports_and_smoke(tmp_path, monkeypatch, runtime_sample, capsys, status):
    work, source = tmp_path / "working", tmp_path / "input"
    source.mkdir()
    oversized = {"messages": [{"role": "user", "content": "x" * 5000}]}
    for split, count in (("train", 3), ("val", 5)):
        (source / f"sft_{split}.jsonl").write_text(
            "".join(json.dumps(sample) + "\n" for sample in [runtime_sample] * count + [oversized])
        )
    monkeypatch.setattr(finetune, "WORK", work)
    monkeypatch.setattr(finetune, "INPUT", source)
    monkeypatch.setattr(finetune, "version", lambda _: "test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "test")
    calls, now = {}, [0]
    monkeypatch.setattr(finetune.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(finetune.subprocess, "run", lambda args, **kwargs: calls.update(pip=args))
    answer = '{"thought":"Done.","tool":"finish","args":{"summary":"Done."}}'

    class Batch(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    class Tokenizer(TinyTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            if kwargs["tokenize"]:
                assert kwargs["enable_thinking"] is False
                assert kwargs["add_generation_prompt"] is True
                return Batch(input_ids=SimpleNamespace(shape=(1, 10)))
            return super().apply_chat_template(*args, **kwargs)

        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))

        def save_pretrained(self, path):
            calls.setdefault("tokenizers", []).append(Path(path).name)

    class Output:
        def __getitem__(self, index):
            assert index == (0, slice(10, None))
            return list(map(ord, answer))

    class Model:
        device = "cuda"

        def save_pretrained(self, path, **kwargs):
            calls["adapter"] = (Path(path).name, kwargs)

        def save_pretrained_merged(self, path, tokenizer, **kwargs):
            calls["merged"] = (Path(path).name, kwargs)

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 512 and kwargs["do_sample"] is False
            calls["smoke"] = calls.get("smoke", 0) + 1
            return Output()

    model, tokenizer = Model(), Tokenizer()

    def load(**kwargs):
        calls["load"] = kwargs
        if status == "expired":
            now[0] = 36000
        return model, tokenizer

    def peft(model, **kwargs):
        calls["peft"] = kwargs
        return model

    def mask(trainer, **kwargs):
        calls["mask"] = kwargs
        return trainer

    class Trainer:
        def __init__(self, **kwargs):
            calls["trainer"] = kwargs
            self.train_dataset = kwargs["train_dataset"]
            self.eval_dataset = kwargs["eval_dataset"]
            self.state = SimpleNamespace(global_step=0, best_metric=None,
                                         best_model_checkpoint=None, log_history=[])

        def train(self):
            calls["train"] = True
            self.state.global_step = 1
            self.state.log_history.append({"loss": 0.5})
            if status == "error":
                raise RuntimeError("training failed")
            self.state.best_metric = 0.5
            self.state.best_model_checkpoint = "checkpoint-1"

    fast = SimpleNamespace(from_pretrained=load, get_peft_model=peft,
                           for_inference=lambda model: None)
    modules = {
        "unsloth": SimpleNamespace(FastLanguageModel=fast),
        "unsloth.chat_templates": SimpleNamespace(train_on_responses_only=mask),
        "torch": SimpleNamespace(float16="float16", inference_mode=nullcontext),
        "datasets": SimpleNamespace(Dataset=SimpleNamespace(
            from_dict=lambda columns: [dict(zip(columns, row, strict=True))
                                       for row in zip(*columns.values(), strict=True)])),
        "transformers": SimpleNamespace(TrainerCallback=object,
                                        DataCollatorForSeq2Seq=lambda **kwargs: kwargs),
        "trl": SimpleNamespace(SFTTrainer=Trainer, SFTConfig=lambda **kwargs: kwargs),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    if status == "error":
        with pytest.raises(RuntimeError, match="training failed"):
            finetune.main()
    else:
        finetune.main()
    assert f"unsloth=={finetune.UNSLOTH_VERSION}" in calls["pip"]
    assert calls["load"]["dtype"] == "float16" and calls["load"]["load_in_4bit"]
    assert calls["load"]["max_seq_length"] == 4096
    assert (calls["peft"]["r"], calls["peft"]["lora_alpha"], calls["peft"]["lora_dropout"]) == (
        32, 64, 0
    )
    assert calls["peft"]["use_gradient_checkpointing"] == "unsloth"
    args = calls["trainer"]["args"]
    for name, expected in {
        "per_device_train_batch_size": 4, "gradient_accumulation_steps": 4,
        "num_train_epochs": 2, "learning_rate": 2e-4, "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03, "fp16": True, "bf16": False, "eval_steps": 200,
        "save_steps": 200, "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
        "greater_is_better": False, "seed": 42, "report_to": "none",
    }.items():
        assert args[name] == expected
    assert calls["mask"]["instruction_part"] == finetune.INSTRUCTION_PART
    assert calls["mask"]["response_part"] == finetune.RESPONSE_PART
    assert calls["adapter"][1]["safe_serialization"] is True
    assert calls["merged"][1]["save_method"] == "merged_16bit"
    assert calls["merged"][1]["safe_serialization"] is None
    assert calls["tokenizers"] == ["adapter", "merged"]
    summary = json.loads((work / "train_summary.json").read_text())
    assert summary["samples"] == {"train": {"used": 3, "dropped": 1},
                                  "val": {"used": 5, "dropped": 1}}
    assert summary["training_returned"] is (status == "completed")
    assert summary["time_limit_reached"] is (status == "expired")
    assert json.loads((work / "trainer_logs.json").read_text()) == (
        [] if status == "expired" else [{"loss": 0.5}]
    )
    assert calls.get("smoke", 0) == (0 if status == "error" else 5)
    output = capsys.readouterr().out
    assert output.count("MASKED ") == 3
    assert output.count("json_step=True") == (0 if status == "error" else 5)
