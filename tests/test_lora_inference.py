from pathlib import Path
import sys

import torch
from torch import nn

from peft import LoraConfig, TaskType, get_peft_model

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
