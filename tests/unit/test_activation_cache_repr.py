"""Regression tests for ActivationCache string representations (issue #1126).

Verifies that formatting an ActivationCache via repr(), str(), f-strings, or IPython's
plain-text pretty printer includes cache key names and never invokes repr(), str(), or
torch operations on cached tensor values, non-tensor entries, or the attached model.
"""

from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache


class _ForbidReprTensor(torch.Tensor):
    """Tensor subclass that fails the test if formatted or operated on during cache repr."""

    op_calls: int = 0
    repr_calls: int = 0
    str_calls: int = 0

    @staticmethod
    def wrap(tensor: torch.Tensor) -> "_ForbidReprTensor":
        return tensor.as_subclass(_ForbidReprTensor)

    @classmethod
    def reset_counts(cls) -> None:
        cls.op_calls = 0
        cls.repr_calls = 0
        cls.str_calls = 0

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        _ForbidReprTensor.repr_calls += 1
        raise AssertionError("ActivationCache must not invoke Tensor.__repr__")

    def __str__(self) -> str:
        _ForbidReprTensor.str_calls += 1
        raise AssertionError("ActivationCache must not invoke Tensor.__str__")

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: Tuple[type, ...],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        cls.op_calls += 1
        return super().__torch_function__(func, types, args, kwargs or {})


class _ForbidReprObject:
    """Sentinel object whose __repr__ and __str__ fail if invoked."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.repr_calls = 0
        self.str_calls = 0

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise AssertionError(f"ActivationCache must not invoke repr() on {self.label}")

    def __str__(self) -> str:
        self.str_calls += 1
        raise AssertionError(f"ActivationCache must not invoke str() on {self.label}")


def _make_forbidden_tensor(*shape: int) -> _ForbidReprTensor:
    return _ForbidReprTensor.wrap(torch.ones(*shape, dtype=torch.float32))


def _format_via_ipython_pretty(cache: ActivationCache) -> Optional[str]:
    try:
        pretty_mod = import_module("IPython.lib.pretty")
    except ImportError:
        return None
    return str(pretty_mod.pretty(cache))


def test_forbid_repr_guards_trip_when_values_are_formatted_directly() -> None:
    """Sanity check: direct value or raw dict formatting must trip the test doubles."""
    tensor = _make_forbidden_tensor(2, 3)
    model_sentinel = _ForbidReprObject("model")
    cache = ActivationCache({"blocks.0.hook_resid_pre": tensor}, model=model_sentinel)

    _ForbidReprTensor.reset_counts()
    with pytest.raises(AssertionError, match="Tensor.__repr__"):
        repr(tensor)
    assert _ForbidReprTensor.repr_calls == 1

    with pytest.raises(AssertionError, match="Tensor.__str__"):
        str(tensor)
    assert _ForbidReprTensor.str_calls == 1

    with pytest.raises(AssertionError, match="Tensor.__repr__"):
        repr(cache.cache_dict)
    assert _ForbidReprTensor.repr_calls == 2

    with pytest.raises(AssertionError, match="model"):
        repr(model_sentinel)
    assert model_sentinel.repr_calls == 1


def test_activation_cache_repr_contains_single_key_name() -> None:
    """Single-entry ActivationCache repr and str include the cache key name."""
    key = "blocks.0.attn.hook_q"
    cache = ActivationCache({key: _make_forbidden_tensor(1, 4, 2, 8)}, model=None)
    _ForbidReprTensor.reset_counts()

    rendered_repr = repr(cache)
    rendered_str = str(cache)

    assert "ActivationCache" in rendered_repr
    assert key in rendered_repr
    assert key in rendered_str
    assert _ForbidReprTensor.repr_calls == 0
    assert _ForbidReprTensor.str_calls == 0
    assert _ForbidReprTensor.op_calls == 0


def test_activation_cache_repr_includes_multiple_key_names_in_order() -> None:
    """Multi-key ActivationCache repr preserves and displays all cache key names."""
    keys: List[str] = [
        "hook_embed",
        "blocks.0.hook_resid_pre",
        "blocks.0.attn.hook_pattern",
        "blocks.0.attn.q.hook_out",
        "blocks.1.mlp.out.hook_out",
        "ln_final.hook_normalized",
    ]
    cache_dict: Dict[str, torch.Tensor] = {k: _make_forbidden_tensor(2, 4, 8) for k in keys}
    cache = ActivationCache(cache_dict, model=None)
    _ForbidReprTensor.reset_counts()

    rendered_repr = repr(cache)
    rendered_str = str(cache)

    for key in keys:
        assert key in rendered_repr
        assert key in rendered_str

    positions = [rendered_repr.index(key) for key in keys]
    assert positions == sorted(positions)
    assert "nonexistent_hook" not in rendered_repr
    assert _ForbidReprTensor.repr_calls == 0
    assert _ForbidReprTensor.op_calls == 0


def test_activation_cache_repr_handles_empty_cache() -> None:
    """Empty ActivationCache formats cleanly without touching model repr."""
    model_sentinel = _ForbidReprObject("model")
    cache = ActivationCache({}, model=model_sentinel)

    rendered = repr(cache)

    assert "ActivationCache" in rendered
    assert model_sentinel.repr_calls == 0
    assert model_sentinel.str_calls == 0


def test_activation_cache_repr_does_not_invoke_cached_tensor_repr_or_ops() -> None:
    """Regression (#1126): formatting ActivationCache never formats or reads cached tensors."""
    cache_dict: Dict[str, torch.Tensor] = {
        "hook_embed": _make_forbidden_tensor(2, 16, 32),
        "blocks.0.attn.hook_scores": _make_forbidden_tensor(2, 4, 16, 16),
        "blocks.0.mlp.hook_post": _make_forbidden_tensor(2, 16, 128),
    }
    cache = ActivationCache(cache_dict, model=None)
    _ForbidReprTensor.reset_counts()

    outputs = [repr(cache), str(cache), f"{cache}", f"{cache!r}"]
    pretty_out = _format_via_ipython_pretty(cache)
    if pretty_out is not None:
        outputs.append(pretty_out)

    for rendered in outputs:
        assert "ActivationCache" in rendered
        for key in cache_dict:
            assert key in rendered

    assert _ForbidReprTensor.repr_calls == 0
    assert _ForbidReprTensor.str_calls == 0
    assert _ForbidReprTensor.op_calls == 0


def test_activation_cache_repr_does_not_invoke_repr_on_model_or_non_tensor_values() -> None:
    """Formatting ActivationCache must not call repr() on cache.model or non-tensor entries."""
    model_sentinel = _ForbidReprObject("model")
    extra_value_sentinel = _ForbidReprObject("cached_non_tensor_value")

    cache_dict: Dict[str, torch.Tensor] = {
        "hook_embed": _make_forbidden_tensor(1, 8, 16),
        "blocks.0.hook_resid_post": _make_forbidden_tensor(1, 8, 16),
    }
    cache = ActivationCache(cache_dict, model=model_sentinel)
    # Guard against repr() on arbitrary non-tensor objects placed in cache_dict post-init.
    cast(Dict[str, Any], cache.cache_dict)["blocks.0.custom_metadata"] = extra_value_sentinel
    _ForbidReprTensor.reset_counts()

    rendered_repr = repr(cache)
    rendered_str = str(cache)
    pretty_out = _format_via_ipython_pretty(cache)

    for key in ("hook_embed", "blocks.0.hook_resid_post", "blocks.0.custom_metadata"):
        assert key in rendered_repr
        assert key in rendered_str
        if pretty_out is not None:
            assert key in pretty_out

    assert model_sentinel.repr_calls == 0
    assert model_sentinel.str_calls == 0
    assert extra_value_sentinel.repr_calls == 0
    assert extra_value_sentinel.str_calls == 0
    assert _ForbidReprTensor.repr_calls == 0
    assert _ForbidReprTensor.str_calls == 0
    assert _ForbidReprTensor.op_calls == 0


def test_activation_cache_repr_scales_to_large_cache_without_value_repr() -> None:
    """Large multi-layer caches include key names without invoking value or model repr."""
    model_sentinel = _ForbidReprObject("model")
    hook_suffixes = (
        "hook_resid_pre",
        "attn.hook_q",
        "attn.hook_k",
        "attn.hook_v",
        "attn.hook_pattern",
        "attn.q.hook_out",
        "mlp.hook_pre",
        "mlp.hook_post",
        "mlp.out.hook_out",
        "hook_resid_post",
    )
    cache_dict: Dict[str, torch.Tensor] = {"hook_embed": _make_forbidden_tensor(1, 4, 8)}
    for layer in range(100):
        for suffix in hook_suffixes:
            cache_dict[f"blocks.{layer}.{suffix}"] = _make_forbidden_tensor(1, 4, 8)

    cache = ActivationCache(cache_dict, model=model_sentinel)
    _ForbidReprTensor.reset_counts()

    rendered = repr(cache)

    assert "ActivationCache" in rendered
    assert "hook_embed" in rendered
    assert "blocks.0.hook_resid_pre" in rendered
    assert "blocks.99.hook_resid_post" in rendered
    assert _ForbidReprTensor.repr_calls == 0
    assert _ForbidReprTensor.str_calls == 0
    assert _ForbidReprTensor.op_calls == 0
    assert model_sentinel.repr_calls == 0
