"""LoRA-aware inference utilities for loading fire-spawned adapters.

This module provides a light-weight orchestration layer for composing a
checkpointed backbone with one or more LoRA adapters at inference time.  The
design goal is to let PERSIST hot-swap evolutionary "fire-spawned" adapters
without retraining or mutating the base weights.  The pipeline supports both
pure PyTorch modules (via a factory/callable) and Hugging Face Hub identifiers.

Typical usage::

    from functools import partial
    from tools.lora_inference import AdapterSpec, LoRAInferencePipeline

    def build_model():
        model = MyBackbone()
        model.load_state_dict(torch.load("backbone.pt"))
        return model

    pipeline = LoRAInferencePipeline(
        base_model=build_model,
        adapter_specs=[AdapterSpec(path="spawned_adapter")],
    )
    model, _ = pipeline.load()
    with torch.no_grad():
        output = model(observation)

When working with Hugging Face transformers you can pass the model identifier
directly (the pipeline will lazily import ``transformers`` and ``peft``)::

    pipeline = LoRAInferencePipeline(
        base_model="gpt2",
        adapter_specs=[AdapterSpec(path="./fire_spawned_gpt2")],
        merge_adapters=True,
    )
    model, tokenizer = pipeline.load()
    print(pipeline.generate("status report"))

The implementation intentionally keeps side effects minimal so it can be safely
bolted into existing evaluation harnesses.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn


logger = logging.getLogger(__name__)


ModuleFactory = Callable[[], nn.Module]
TokenizerFactory = Callable[[], Any]


def _resolve_dtype(value: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    """Convert a ``torch.dtype`` specification to a concrete dtype."""

    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        if not hasattr(torch, value):
            raise ValueError(f"Unrecognised torch dtype string: {value}")
        resolved = getattr(torch, value)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Attribute '{value}' is not a torch.dtype")
        return resolved
    raise TypeError(f"Unsupported dtype specification: {value!r}")


def _require_module(module_name: str, error_message: str):
    """Import ``module_name`` if available, otherwise raise ``RuntimeError``."""

    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(error_message)
    return importlib.import_module(module_name)


@dataclass
class AdapterSpec:
    """Configuration for an individual LoRA adapter."""

    path: str
    name: Optional[str] = None
    weight: Optional[float] = None
    load_kwargs: Dict[str, Any] = field(default_factory=dict)


class LoRAInferencePipeline:
    """Compose a backbone checkpoint with one or more LoRA adapters at runtime."""

    def __init__(
        self,
        *,
        base_model: Union[str, nn.Module, ModuleFactory],
        base_checkpoint: Optional[str] = None,
        adapter_specs: Optional[Sequence[AdapterSpec]] = None,
        tokenizer: Optional[Union[str, Any, TokenizerFactory]] = None,
        base_model_class: Optional[type] = None,
        tokenizer_class: Optional[type] = None,
        base_model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        strict_loading: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        merge_adapters: bool = False,
        merged_adapter_name: str = "merged",
    ) -> None:
        self._base_model_source = base_model
        self._base_checkpoint = base_checkpoint
        self._adapter_specs: Tuple[AdapterSpec, ...] = tuple(adapter_specs or ())
        self._tokenizer_source = tokenizer
        self._base_model_class = base_model_class
        self._tokenizer_class = tokenizer_class
        self._base_model_kwargs = dict(base_model_kwargs or {})
        self._tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self._dtype = _resolve_dtype(torch_dtype)
        self._device = torch.device(device) if device is not None else None
        self._strict_loading = strict_loading
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._trust_remote_code = trust_remote_code
        self._merge_adapters = merge_adapters
        self._merged_adapter_name = merged_adapter_name

        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.loaded_adapter_names: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, force_reload: bool = False) -> Tuple[nn.Module, Optional[Any]]:
        """Load the backbone and attach any configured adapters."""

        if self.model is not None and not force_reload:
            return self.model, self.tokenizer

        model = self._instantiate_base_model()
        self._load_checkpoint_if_available(model)

        if self._adapter_specs:
            model = self._attach_adapters(model)

        model = self._finalise_device(model)
        tokenizer = self._instantiate_tokenizer()

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def forward(self, *args, **kwargs):
        """Run a forward pass with gradients disabled."""

        if self.model is None:
            raise RuntimeError("Call 'load()' before executing inference.")
        self.model.eval()
        with torch.no_grad():
            return self.model(*args, **kwargs)

    __call__ = forward

    def generate(self, prompt: Union[str, Sequence[str]], **generate_kwargs):
        """Generate text using the loaded model and tokenizer."""

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Text generation requires a loaded model and tokenizer.")

        encoded = self.tokenizer(prompt, return_tensors="pt", **self._tokenizer_kwargs)
        target_device = self._inference_device(self.model)
        encoded = {k: v.to(target_device) for k, v in encoded.items()}
        self.model.eval()
        with torch.no_grad():
            output_tokens = self.model.generate(**encoded, **generate_kwargs)
        if isinstance(prompt, str):
            return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_tokens
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _instantiate_base_model(self) -> nn.Module:
        source = self._base_model_source
        if isinstance(source, nn.Module):
            return source
        if callable(source):
            model = source()
            if not isinstance(model, nn.Module):
                raise TypeError("Base model factory must return a torch.nn.Module")
            return model
        if isinstance(source, str):
            transformers_module = _require_module(
                "transformers",
                "transformers must be installed to load models by identifier.",
            )
            model_cls = self._base_model_class or getattr(transformers_module, "AutoModelForCausalLM")
            kwargs = dict(self._base_model_kwargs)
            if self._dtype is not None and "torch_dtype" not in kwargs:
                kwargs["torch_dtype"] = self._dtype
            if self._load_in_8bit:
                kwargs.setdefault("load_in_8bit", True)
            if self._load_in_4bit:
                kwargs.setdefault("load_in_4bit", True)
            if "trust_remote_code" not in kwargs:
                kwargs["trust_remote_code"] = self._trust_remote_code
            model = model_cls.from_pretrained(source, **kwargs)
            return model
        raise TypeError(f"Unsupported base model specification: {source!r}")

    def _load_checkpoint_if_available(self, model: nn.Module) -> None:
        if self._base_checkpoint is None:
            return
        state = torch.load(self._base_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if self._strict_loading and (missing or unexpected):
            raise RuntimeError(
                "Checkpoint loading failed: "
                f"missing_keys={list(missing)}, unexpected_keys={list(unexpected)}"
            )
        if missing:
            logger.warning("Missing keys during checkpoint load: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys during checkpoint load: %s", unexpected)

    def _attach_adapters(self, model: nn.Module) -> nn.Module:
        peft_module = _require_module(
            "peft",
            "peft must be installed to attach LoRA adapters.",
        )
        peft_model_cls = getattr(peft_module, "PeftModel")

        adapter_names: Tuple[str, ...] = ()
        peft_model: Optional[nn.Module] = None
        for idx, spec in enumerate(self._adapter_specs):
            adapter_name = spec.name or f"adapter_{idx}"
            load_kwargs = dict(spec.load_kwargs)
            load_kwargs.setdefault("is_trainable", False)
            if peft_model is None:
                peft_model = peft_model_cls.from_pretrained(
                    model,
                    spec.path,
                    adapter_name=adapter_name,
                    **load_kwargs,
                )
            else:
                peft_model.load_adapter(
                    spec.path,
                    adapter_name=adapter_name,
                    **load_kwargs,
                )
            adapter_names = (*adapter_names, adapter_name)

        self.loaded_adapter_names = tuple(adapter_names)
        assert peft_model is not None  # for mypy, adapters exist here

        # Combine adapters when requested.
        if len(adapter_names) > 1:
            weights = [spec.weight if spec.weight is not None else 1.0 for spec in self._adapter_specs]
            if any(weight != 1.0 for weight in weights):
                peft_model.add_weighted_adapter(
                    list(adapter_names),
                    weights,
                    adapter_name=self._merged_adapter_name,
                )
                peft_model.set_adapter(self._merged_adapter_name)
            elif self._merge_adapters:
                peft_model.add_weighted_adapter(
                    list(adapter_names),
                    [1.0] * len(adapter_names),
                    adapter_name=self._merged_adapter_name,
                )
                peft_model.set_adapter(self._merged_adapter_name)
            else:
                peft_model.set_adapter(adapter_names[-1])
        else:
            peft_model.set_adapter(adapter_names[-1])

        if self._merge_adapters and not (self._load_in_8bit or self._load_in_4bit):
            merged_model = peft_model.merge_and_unload()
            return merged_model

        return peft_model

    def _instantiate_tokenizer(self) -> Optional[Any]:
        source = self._tokenizer_source
        if source is None:
            return None
        if callable(source):
            return source()
        if isinstance(source, str):
            transformers_module = _require_module(
                "transformers",
                "transformers must be installed to load tokenizers by identifier.",
            )
            tokenizer_cls = self._tokenizer_class or getattr(transformers_module, "AutoTokenizer")
            kwargs = dict(self._tokenizer_kwargs)
            return tokenizer_cls.from_pretrained(source, **kwargs)
        return source

    def _finalise_device(self, model: nn.Module) -> nn.Module:
        if self._device is None:
            return model
        if hasattr(model, "to"):
            model = model.to(self._device)
        return model

    @staticmethod
    def _inference_device(model: nn.Module) -> torch.device:
        if hasattr(model, "device"):
            return torch.device(model.device)
        first_param = next(model.parameters(), None)
        if first_param is not None:
            return first_param.device
        return torch.device("cpu")

