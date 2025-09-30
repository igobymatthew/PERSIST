from pathlib import Path
import sys

import pytest
import torch
from torch import nn

try:  # pragma: no cover - simple availability guard
    from peft import LoraConfig, TaskType, get_peft_model
except ModuleNotFoundError:  # pragma: no cover
    LoraConfig = TaskType = get_peft_model = None  # type: ignore[assignment]
    PEFT_AVAILABLE = False
else:  # pragma: no cover
    PEFT_AVAILABLE = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lora_inference import AdapterSpec, LoRAInferencePipeline


def _make_base_model() -> nn.Sequential:
    layer = nn.Linear(4, 4, bias=False)
    layer.weight.data.copy_(torch.eye(4))
    return nn.Sequential(layer)


def _create_lora_adapter(path, fill_a: float, fill_b: float) -> None:
    base = _make_base_model()
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["0"],
        inference_mode=False,
        r=2,
        lora_alpha=2,
    )
    peft_model = get_peft_model(base, config)
    peft_model.base_model.model[0].lora_A["default"].weight.data.fill_(fill_a)
    peft_model.base_model.model[0].lora_B["default"].weight.data.fill_(fill_b)
    peft_model.save_pretrained(str(path))


@pytest.mark.skipif(not PEFT_AVAILABLE, reason="peft is required for adapter tests")
def test_pipeline_merges_single_adapter(tmp_path):
    base_path = tmp_path / "backbone.pt"
    torch.save(_make_base_model().state_dict(), base_path)

    adapter_dir = tmp_path / "adapter_single"
    _create_lora_adapter(adapter_dir, fill_a=0.5, fill_b=0.25)

    pipeline = LoRAInferencePipeline(
        base_model=_make_base_model,
        base_checkpoint=str(base_path),
        adapter_specs=[AdapterSpec(path=str(adapter_dir))],
        merge_adapters=True,
        device="cpu",
    )
    model, _ = pipeline.load()

    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        output = model(inputs)

    expected = torch.tensor([[3.5, 4.5, 5.5, 6.5]])
    assert torch.allclose(output, expected)
    assert pipeline.loaded_adapter_names == ("adapter_0",)


@pytest.mark.skipif(not PEFT_AVAILABLE, reason="peft is required for adapter tests")
def test_pipeline_combines_multiple_adapters(tmp_path):
    base_path = tmp_path / "backbone.pt"
    torch.save(_make_base_model().state_dict(), base_path)

    adapter_a = tmp_path / "adapter_a"
    adapter_b = tmp_path / "adapter_b"
    _create_lora_adapter(adapter_a, fill_a=0.5, fill_b=0.25)
    _create_lora_adapter(adapter_b, fill_a=1.0, fill_b=0.5)

    pipeline = LoRAInferencePipeline(
        base_model=_make_base_model,
        base_checkpoint=str(base_path),
        adapter_specs=[
            AdapterSpec(path=str(adapter_a)),
            AdapterSpec(path=str(adapter_b)),
        ],
        merge_adapters=True,
        merged_adapter_name="combo",
        device="cpu",
    )

    model, _ = pipeline.load()
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        output = pipeline(inputs)

    expected = torch.tensor([[13.5, 14.5, 15.5, 16.5]])
    assert torch.allclose(output, expected)
    assert pipeline.loaded_adapter_names == ("adapter_0", "adapter_1")


def test_generate_respects_tokenizer_init_kwargs():
    class DummyTokenizer:
        def __init__(self, padding_side: str = "right"):
            self.padding_side = padding_side

        def __call__(self, prompt, return_tensors="pt", **kwargs):
            if "padding_side" in kwargs:
                raise AssertionError("padding_side should not be passed at call time")
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            input_ids = torch.ones(batch_size, 1, dtype=torch.long)
            return {"input_ids": input_ids}

        def decode(self, ids, skip_special_tokens=True):
            return "decoded"

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, *args, **kwargs):
            raise NotImplementedError

        def generate(self, input_ids, **kwargs):
            return torch.ones_like(input_ids)

    tokenizer_instance = DummyTokenizer(padding_side="left")
    pipeline = LoRAInferencePipeline(
        base_model=DummyModel(),
        tokenizer=lambda: tokenizer_instance,
        tokenizer_kwargs={"padding_side": "left"},
    )

    pipeline.load()
    assert pipeline.generate("prompt") == "decoded"
