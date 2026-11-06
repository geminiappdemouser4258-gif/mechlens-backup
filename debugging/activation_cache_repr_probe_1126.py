"""Diagnostic probe for issue #1126: ActivationCache display vs. adjacent slow paths.

Separates and times each path that could be blamed for a notebook hang:
1. Isolated ActivationCache.__repr__ and str() (O(num_keys), touches string keys only).
2. IPython DisplayFormatter / PlainTextFormatter cell rendering (when IPython is installed).
3. Unpacked (logits, cache) tuple repr from a bare model.run_with_cache(...) cell expression.
4. Raw dict[str, Tensor] repr (cache.cache_dict, dict(cache), or return_cache_object=False).
5. IDE variable-inspector style .items() traversal with per-tensor host readback.
6. Optional live model capture vs. display split on TransformerBridge or HookedTransformer.

Usage (from repo root):
    uv run python debugging/activation_cache_repr_probe_1126.py
    uv run python debugging/activation_cache_repr_probe_1126.py --device cuda --n-layers 80
    uv run python debugging/activation_cache_repr_probe_1126.py --mode bridge --model distilgpt2
"""

from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from transformer_lens.ActivationCache import ActivationCache

_STANDARD_BLOCK_HOOKS: Tuple[str, ...] = (
    "hook_resid_pre",
    "hook_resid_mid",
    "hook_resid_post",
    "ln1.hook_scale",
    "ln1.hook_normalized",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "attn.hook_z",
    "attn.hook_attn_scores",
    "attn.hook_pattern",
    "hook_attn_out",
    "ln2.hook_scale",
    "ln2.hook_normalized",
    "mlp.hook_pre",
    "mlp.hook_post",
    "hook_mlp_out",
)

_BRIDGE_SUBMODULE_HOOKS: Tuple[str, ...] = (
    "attn.q.hook_in",
    "attn.q.hook_out",
    "attn.k.hook_in",
    "attn.k.hook_out",
    "attn.v.hook_in",
    "attn.v.hook_out",
    "attn.o.hook_in",
    "attn.o.hook_out",
    "mlp.gate.hook_in",
    "mlp.gate.hook_out",
    "mlp.in.hook_in",
    "mlp.in.hook_out",
    "mlp.out.hook_in",
    "mlp.out.hook_out",
)


class _InstrumentedTensor(torch.Tensor):
    """Tracks whether formatting invokes Tensor.__repr__ or any torch operation."""

    op_calls: int = 0
    repr_calls: int = 0

    @staticmethod
    def wrap(tensor: torch.Tensor) -> "_InstrumentedTensor":
        return tensor.as_subclass(_InstrumentedTensor)

    @classmethod
    def reset_counts(cls) -> None:
        cls.op_calls = 0
        cls.repr_calls = 0

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        _InstrumentedTensor.repr_calls += 1
        base = self.as_subclass(torch.Tensor)
        if tensor_contents is not None:
            return base.__repr__(tensor_contents=tensor_contents)
        return repr(base)

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


@dataclass(frozen=True)
class ProbeTiming:
    """Single probe measurement with tensor-touch counters and output size."""

    name: str
    elapsed_ms: float
    output_chars: int
    tensor_repr_calls: int
    tensor_op_calls: int
    note: str = ""


def _sync_device(device: torch.device) -> None:
    """Flush pending async kernels so capture cost never bleeds into print timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _measure_step(
    name: str,
    device: torch.device,
    fn: Callable[[], str],
    note: str = "",
) -> ProbeTiming:
    _sync_device(device)
    _InstrumentedTensor.reset_counts()
    start = time.perf_counter()
    rendered = fn()
    _sync_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ProbeTiming(
        name=name,
        elapsed_ms=elapsed_ms,
        output_chars=len(rendered),
        tensor_repr_calls=_InstrumentedTensor.repr_calls,
        tensor_op_calls=_InstrumentedTensor.op_calls,
        note=note,
    )


def build_synthetic_cache(
    *,
    n_layers: int = 32,
    batch_size: int = 2,
    seq_len: int = 128,
    d_model: int = 512,
    n_heads: int = 8,
    d_vocab: int = 50257,
    include_bridge_hooks: bool = True,
    device_str: str = "cpu",
) -> Tuple[torch.Tensor, ActivationCache, torch.device]:
    """Build an instrumented (logits, ActivationCache) pair without loading model weights."""
    device = torch.device(device_str)
    d_head = max(1, d_model // n_heads)
    d_mlp = d_model * 4

    cache_dict: Dict[str, torch.Tensor] = {
        "hook_embed": _InstrumentedTensor.wrap(
            torch.randn(batch_size, seq_len, d_model, device=device)
        ),
        "hook_pos_embed": _InstrumentedTensor.wrap(
            torch.randn(batch_size, seq_len, d_model, device=device)
        ),
    }

    hook_names: List[str] = list(_STANDARD_BLOCK_HOOKS)
    if include_bridge_hooks:
        hook_names.extend(_BRIDGE_SUBMODULE_HOOKS)

    for layer in range(n_layers):
        for suffix in hook_names:
            key = f"blocks.{layer}.{suffix}"
            if "pattern" in suffix or "attn_scores" in suffix:
                raw = torch.randn(batch_size, n_heads, seq_len, seq_len, device=device)
            elif suffix.endswith((".hook_q", ".hook_k", ".hook_v", ".hook_z")):
                raw = torch.randn(batch_size, seq_len, n_heads, d_head, device=device)
            elif "scale" in suffix:
                raw = torch.randn(batch_size, seq_len, 1, device=device)
            elif "mlp." in suffix and not suffix.endswith("mlp.out.hook_out"):
                raw = torch.randn(batch_size, seq_len, d_mlp, device=device)
            else:
                raw = torch.randn(batch_size, seq_len, d_model, device=device)
            cache_dict[key] = _InstrumentedTensor.wrap(raw)

    cache_dict["ln_final.hook_scale"] = _InstrumentedTensor.wrap(
        torch.randn(batch_size, seq_len, 1, device=device)
    )
    cache_dict["ln_final.hook_normalized"] = _InstrumentedTensor.wrap(
        torch.randn(batch_size, seq_len, d_model, device=device)
    )

    logits = _InstrumentedTensor.wrap(torch.randn(batch_size, seq_len, d_vocab, device=device))
    cache = ActivationCache(cache_dict, model=None, has_batch_dim=True)
    return logits, cache, device


def _try_ipython_format(obj: Any) -> Optional[Tuple[str, List[str]]]:
    """Format obj via IPython's DisplayFormatter when IPython is available."""
    try:
        formatters_mod = importlib.import_module("IPython.core.formatters")
    except ImportError:
        return None
    formatter = formatters_mod.DisplayFormatter()
    data, _metadata = formatter.format(obj)
    mime_keys = [str(k) for k in data.keys()]
    plain = str(data.get("text/plain", ""))
    return plain, mime_keys


def _estimate_cache_megabytes(cache: ActivationCache) -> float:
    total_bytes = 0
    for val in cache.cache_dict.values():
        if isinstance(val, torch.Tensor):
            total_bytes += val.numel() * val.element_size()
    return total_bytes / (1024.0 * 1024.0)


def run_display_probes(
    logits: torch.Tensor,
    cache: ActivationCache,
    device: torch.device,
    *,
    dict_sample_keys: int = 8,
    full_dict_repr: bool = False,
) -> List[ProbeTiming]:
    """Run isolated display/formatting probes against an existing (logits, cache) pair."""
    results: List[ProbeTiming] = []

    results.append(
        _measure_step(
            "1. repr(cache) [ActivationCache.__repr__]",
            device,
            lambda: repr(cache),
            note="keys-only path in ActivationCache.py",
        )
    )
    results.append(
        _measure_step(
            "2. str(cache) [print(cache) payload]",
            device,
            lambda: str(cache),
            note="falls back to ActivationCache.__repr__",
        )
    )

    ipy_probe = _try_ipython_format(cache)
    if ipy_probe is not None:
        results.append(
            _measure_step(
                "3. IPython DisplayFormatter.format(cache)",
                device,
                lambda: (_try_ipython_format(cache) or ("", []))[0],
                note=f"MIME types: {', '.join(ipy_probe[1])}",
            )
        )
    else:
        results.append(
            ProbeTiming(
                name="3. IPython DisplayFormatter.format(cache)",
                elapsed_ms=0.0,
                output_chars=0,
                tensor_repr_calls=0,
                tensor_op_calls=0,
                note="skipped (IPython not installed in this environment)",
            )
        )

    results.append(
        _measure_step(
            "4. repr((logits, cache)) [unpacked cell tuple]",
            device,
            lambda: repr((logits, cache)),
            note="formats logits via Tensor.__repr__ before cache",
        )
    )

    all_keys = list(cache.cache_dict.keys())
    if full_dict_repr:
        results.append(
            _measure_step(
                f"5. repr(cache.cache_dict) [all {len(all_keys)} tensors]",
                device,
                lambda: repr(cache.cache_dict),
                note="return_cache_object=False / raw dict path",
            )
        )
    else:
        sample_n = max(1, min(dict_sample_keys, len(all_keys)))
        sampled_dict = {k: cache.cache_dict[k] for k in all_keys[:sample_n]}
        sample_timing = _measure_step(
            f"5. repr(cache.cache_dict[:{sample_n}]) [sampled raw dict]",
            device,
            lambda: repr(sampled_dict),
            note="",
        )
        scale = len(all_keys) / float(sample_n)
        est_ms = sample_timing.elapsed_ms * scale
        est_mb = (sample_timing.output_chars * scale) / (1024.0 * 1024.0)
        results.append(
            ProbeTiming(
                name=sample_timing.name,
                elapsed_ms=sample_timing.elapsed_ms,
                output_chars=sample_timing.output_chars,
                tensor_repr_calls=sample_timing.tensor_repr_calls,
                tensor_op_calls=sample_timing.tensor_op_calls,
                note=f"linearly ~{est_ms:,.0f} ms & ~{est_mb:.1f} MB for all {len(all_keys)} keys",
            )
        )

    def _simulate_variable_inspector() -> str:
        summaries: List[str] = []
        for key, tensor in cache.items():
            if isinstance(tensor, torch.Tensor):
                first_val = float(tensor.flatten()[0].detach().cpu().item())
                summaries.append(f"{key}:{tuple(tensor.shape)}:{first_val:.3f}")
        return "\n".join(summaries)

    results.append(
        _measure_step(
            "6. IDE variable-inspector walk (cache.items() + item())",
            device,
            _simulate_variable_inspector,
            note="per-tensor host sync if a notebook inspector walks items()",
        )
    )

    return results


def run_live_model_probe(
    *,
    mode: str,
    model_name: str,
    device_str: str,
    seq_len: int,
    compat_mode: bool,
    cache_device: Optional[str],
    dict_sample_keys: int,
    full_dict_repr: bool,
) -> Tuple[ProbeTiming, ActivationCache, List[ProbeTiming]]:
    """Run live run_with_cache capture and separate capture latency from display latency."""
    device = torch.device(device_str)
    tokens = torch.ones((1, seq_len), dtype=torch.long, device=device)

    if mode == "bridge":
        bridge_mod = importlib.import_module("transformer_lens.model_bridge")
        model = bridge_mod.TransformerBridge.boot_transformers(model_name, device=device_str)
        if compat_mode:
            model.enable_compatibility_mode()
    elif mode == "hooked-transformer":
        ht_mod = importlib.import_module("transformer_lens")
        model = ht_mod.HookedTransformer.from_pretrained(model_name, device=device_str)
    else:
        raise ValueError(f"Unsupported live mode: {mode}")

    cache_kwargs: Dict[str, Any] = {}
    if cache_device is not None:
        cache_kwargs["device"] = cache_device

    _sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, raw_cache = model.run_with_cache(tokens, **cache_kwargs)
    _sync_device(device)
    capture_ms = (time.perf_counter() - t0) * 1000.0

    if not isinstance(raw_cache, ActivationCache):
        raise RuntimeError(f"Expected ActivationCache, got {type(raw_cache).__name__}")

    instrumented_dict: Dict[str, torch.Tensor] = {
        k: _InstrumentedTensor.wrap(v) if isinstance(v, torch.Tensor) else v
        for k, v in raw_cache.cache_dict.items()
    }
    instrumented_cache = ActivationCache(
        instrumented_dict,
        model=raw_cache.model,
        has_batch_dim=raw_cache.has_batch_dim,
    )
    instrumented_logits = (
        _InstrumentedTensor.wrap(logits) if isinstance(logits, torch.Tensor) else logits
    )

    capture_timing = ProbeTiming(
        name=f"0. {mode}.run_with_cache(seq_len={seq_len})",
        elapsed_ms=capture_ms,
        output_chars=0,
        tensor_repr_calls=0,
        tensor_op_calls=0,
        note=f"compat_mode={compat_mode}, cache_device={cache_device or 'model'}",
    )
    probes = run_display_probes(
        instrumented_logits,
        instrumented_cache,
        device,
        dict_sample_keys=dict_sample_keys,
        full_dict_repr=full_dict_repr,
    )
    return capture_timing, instrumented_cache, probes


def print_probe_report(
    *,
    header: str,
    cache: ActivationCache,
    timings: Sequence[ProbeTiming],
) -> None:
    """Print a compact diagnostic table comparing all candidate paths."""
    cache_mb = _estimate_cache_megabytes(cache)
    print(f"\n=== {header} ===")
    print(f"Cached keys: {len(cache):,} | Footprint: {cache_mb:,.2f} MiB")
    print(
        f"{'Path':<54} {'Time (ms)':>11} {'Out Chars':>12} "
        f"{'Tensor repr':>12} {'Torch ops':>10}  Note"
    )
    print("-" * 124)
    for row in timings:
        print(
            f"{row.name:<54} {row.elapsed_ms:>11.3f} {row.output_chars:>12,} "
            f"{row.tensor_repr_calls:>12,} {row.tensor_op_calls:>10,}  {row.note}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic timing probe for issue #1126 (ActivationCache repr vs. slow paths)."
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "bridge", "hooked-transformer"),
        default="synthetic",
        help="Use synthetic tensors (no downloads) or run a live model forward pass.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target torch device (e.g. cpu, cuda, mps).",
    )
    parser.add_argument("--model", default="distilgpt2", help="Model ID for live modes.")
    parser.add_argument("--n-layers", type=int, default=32, help="Layers in synthetic mode.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size in synthetic mode.")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length.")
    parser.add_argument("--d-model", type=int, default=512, help="Hidden size in synthetic mode.")
    parser.add_argument("--n-heads", type=int, default=8, help="Head count in synthetic mode.")
    parser.add_argument(
        "--compat-mode",
        action="store_true",
        help="Enable compatibility_mode on TransformerBridge in --mode bridge.",
    )
    parser.add_argument(
        "--cache-device",
        default=None,
        help="Optional offload device passed to run_with_cache(device=...).",
    )
    parser.add_argument(
        "--dict-sample-keys",
        type=int,
        default=8,
        help="Number of keys to sample for raw dict repr timing when --full-dict-repr is off.",
    )
    parser.add_argument(
        "--full-dict-repr",
        action="store_true",
        help="Run repr(cache.cache_dict) across all keys (slow; emits multi-MB text).",
    )
    return parser.parse_args(argv)


def run_probe(argv: Optional[Sequence[str]] = None) -> List[ProbeTiming]:
    """Entry point callable from both CLI and a notebook cell (e.g. run_probe([]))."""
    args = parse_args(argv)
    if args.mode == "synthetic":
        logits, cache, device = build_synthetic_cache(
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            device_str=args.device,
        )
        timings = run_display_probes(
            logits,
            cache,
            device,
            dict_sample_keys=args.dict_sample_keys,
            full_dict_repr=args.full_dict_repr,
        )
        print_probe_report(
            header=f"Issue #1126 Synthetic Probe (device={device}, layers={args.n_layers})",
            cache=cache,
            timings=timings,
        )
        return timings

    capture_timing, live_cache, live_probes = run_live_model_probe(
        mode=args.mode,
        model_name=args.model,
        device_str=args.device,
        seq_len=args.seq_len,
        compat_mode=args.compat_mode,
        cache_device=args.cache_device,
        dict_sample_keys=args.dict_sample_keys,
        full_dict_repr=args.full_dict_repr,
    )
    all_timings = [capture_timing, *live_probes]
    print_probe_report(
        header=f"Issue #1126 Live Probe ({args.mode}: {args.model}, device={args.device})",
        cache=live_cache,
        timings=all_timings,
    )
    return all_timings


if __name__ == "__main__":
    run_probe()
