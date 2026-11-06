# Issue #1126 Triage: `ActivationCache` Representation vs. Notebook Hang Paths

| Field | Value |
| :--- | :--- |
| **Upstream Issue** | `#1126` — *bug: ActivationCache repr is very slow for large caches* |
| **Reporter & Date** | `@repr-rob` (opened `2026-07-08`) |
| **Recommended Label** | `needs-repro` (hold `bug` until a minimal reproducer isolates the slow path) |
| **Investigation Branch** | [`investigate/activation-cache-repr-1126`](https://github.com/geminiappdemouser4258-gif/mechlens-backup/tree/investigate/activation-cache-repr-1126) |
| **Diagnostic Probe** | [`debugging/activation_cache_repr_probe_1126.py`](https://github.com/geminiappdemouser4258-gif/mechlens-backup/blob/investigate/activation-cache-repr-1126/debugging/activation_cache_repr_probe_1126.py) |
| **Unit Regression Guard** | [`tests/unit/test_activation_cache_repr.py`](https://github.com/geminiappdemouser4258-gif/mechlens-backup/blob/investigate/activation-cache-repr-1126/tests/unit/test_activation_cache_repr.py) |
| **Production Files Touched** | None (`transformer_lens/ActivationCache.py` intentionally unchanged pending repro) |

## 1. Issue Premise vs. Current Codebase Facts

Verbatim report from `@repr-rob` in `#1126`:

> Printing a large cache hangs the notebook. Could repr be truncated?

Before changing `ActivationCache.__repr__` (which researchers rely on to inspect available hook names), we need to keep what the current checkout **proves** separate from what `#1126` **assumes**.

| Dimension | Proven in Current Checkout | Unproven Assumption in `#1126` |
| :--- | :--- | :--- |
| **What `repr(cache)` executes** | `ActivationCache.__repr__` (`transformer_lens/ActivationCache.py:176-183`) runs `f"ActivationCache with keys {list(self.cache_dict.keys())}"`. It inherits from `object` (not `dict`) and defines no `__str__`, `_repr_pretty_`, or `_repr_html_`. | Assumes `ActivationCache.__repr__` formats or traverses cached activation tensors (similar to `dict.__repr__`). |
| **Tensor & device work during formatting** | `repr(cache)`, `str(cache)`, and IPython's `PlainTextFormatter` invoke **0** `Tensor.__repr__` calls, **0** `torch` ops, **0** CUDA/MPS host syncs, and **0** `repr(cache.model)` calls (guarded in [`tests/unit/test_activation_cache_repr.py`](https://github.com/geminiappdemouser4258-gif/mechlens-backup/blob/investigate/activation-cache-repr-1126/tests/unit/test_activation_cache_repr.py)). | Assumes Python-side `repr(cache)` computation is expensive enough to freeze a kernel on large models. |
| **Output payload size** | Even on an 80-layer model with ~31 hooks/layer (~2,484 keys), `repr(cache)` emits ~75–90 KB of plain ASCII hook names in a single string. | Assumes truncating the key list in `ActivationCache.__repr__` would resolve the user's hang. |
| **What object was actually displayed** | `HookedTransformer.run_with_cache` (`HookedTransformer.py:738-742`) and `TransformerBridge.run_with_cache` (`bridge.py:2235-2240`) return `(output, cache)` — or `(output, cache_dict)` when `return_cache_object=False` — while `HookedRootModule.run_with_cache` (`HookedRootModule.py:435`) always returns a raw `dict`. | Does not specify whether the cell evaluated `cache`, `model.run_with_cache(...)` without unpacking, `cache.cache_dict`, or `return_cache_object=False`, nor which notebook frontend was used. |

## 2. Candidate Bottlenecks Worth Reproducing First

Ranked from highest likelihood of causing a multi-second notebook freeze to lowest:

| Rank | Candidate Path | Exact Code Location | Why It Hangs a Notebook | How to Confirm |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Unpacked `(logits, cache)` cell expression** | `HookedTransformer.py:740`, `bridge.py:2239` | Evaluating `model.run_with_cache(tokens)` as the final line of a cell formats the 2-tuple `(logits, cache)`, triggering `torch.Tensor.__repr__` + device-to-host sync on `[batch, pos, d_vocab]` `logits` before `cache` is formatted. | Compare Probe Step 4 (`repr((logits, cache))`) against Step 1 (`repr(cache)`). |
| **2** | **Raw `dict[str, Tensor]` instead of `ActivationCache`** | `HookedTransformer.py:742`, `HookedRootModule.py:435`, `bridge.py:2240` | Passing `return_cache_object=False`, calling `HookedRootModule.run_with_cache`, or printing `cache.cache_dict` / `dict(cache)` invokes `Tensor.__repr__` and host readback on **every** cached tensor, emitting tens of megabytes of text. | Compare Probe Step 5 (`repr(cache.cache_dict)`) against Step 1 (`repr(cache)`). |
| **3** | **Capture-time sync or attention materialization** | `bridge.py:2113`, `bridge.py:2192-2193`, `bridge.py:2212-2234`, `HookedRootModule.py:393-395` | `TransformerBridge.run_with_cache` sets `output_attentions=True` by default, `compatibility_mode=True` expands legacy hook aliases post-forward, and `device="cpu"` runs per-hook `.detach().to(cache_device)` syncs; async GPU work also flushes at cell print time if unsynchronized. | Compare Probe Step 0 (`run_with_cache` with explicit CUDA/MPS sync) against Step 1 (`repr(cache)`). |
| **4** | **IDE variable explorer / data viewer inspection** | `ActivationCache.py:147`, `ActivationCache.py:291-297` | Notebook variable panes (VS Code, JupyterLab, Colab) that auto-inspect workspace mappings after each cell can iterate `cache.items()` or walk `cache.model` (`nn.Module`), triggering per-tensor host readbacks outside `ActivationCache.__repr__`. | Compare Probe Step 6 (`cache.items()` + `.item()` walk) and test reproducing in plain `ipython` vs. the IDE notebook UI. |
| **5** | **Frontend DOM stall on a single 80 KB unbroken line** | `ActivationCache.py:183` | `list(self.cache_dict.keys())` renders all 1,000–2,500 keys on **one unbroken line** (`ActivationCache with keys [...]`). Python finishes in sub-millisecond time, though browser DOM soft-wrapping or regex linkifiers could stutter on very wide single-line outputs. | Render `print(cache)` on a 2,500-key synthetic cache inside JupyterLab, VS Code Notebooks, and Colab. |

## 3. Diagnostic Measurements (Pending Local Execution)

Run the committed unit guards and diagnostic probe from the repository root on `investigate/activation-cache-repr-1126`:

```bash
# 1. Verify unit regression guards pass
uv run pytest tests/unit/test_activation_cache_repr.py

# 2. Isolate synthetic 80-layer display paths (no model download required)
uv run python debugging/activation_cache_repr_probe_1126.py --mode synthetic --n-layers 80

# 3. Separate live capture latency from display latency on TransformerBridge & HookedTransformer
uv run python debugging/activation_cache_repr_probe_1126.py --mode bridge --model distilgpt2
uv run python debugging/activation_cache_repr_probe_1126.py --mode bridge --model distilgpt2 --compat-mode
uv run python debugging/activation_cache_repr_probe_1126.py --mode hooked-transformer --model distilgpt2
```

### 3.1 Environment Metadata

| Parameter | Recorded Value |
| :--- | :--- |
| **Commit / Branch** | `investigate/activation-cache-repr-1126` ([UNFILLED — confirm HEAD SHA]) |
| **Hardware (CPU / Accelerator)** | *Pending* (`[UNFILLED]`) |
| **OS & Python Version** | *Pending* (`[UNFILLED]`) |
| **`torch` & `transformers` Versions** | *Pending* (`[UNFILLED]`) |
| **`IPython` / Notebook Frontend** | *Pending* (`[UNFILLED]`) |

### 3.2 Synthetic Display-Path Timings (`--mode synthetic --n-layers 80`)

| Probe Step | Elapsed (ms) | Output Chars | `Tensor.__repr__` Calls | `torch` Ops | Status / Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. `repr(cache)` (`ActivationCache.__repr__`)** | *Pending* | *Pending* | *Pending* (expected `0`) | *Pending* (expected `0`) | `[UNFILLED — run probe]` |
| **2. `str(cache)` (`print(cache)` payload)** | *Pending* | *Pending* | *Pending* (expected `0`) | *Pending* (expected `0`) | `[UNFILLED — run probe]` |
| **3. `IPython DisplayFormatter.format(cache)`** | *Pending* | *Pending* | *Pending* (expected `0`) | *Pending* (expected `0`) | `[UNFILLED — run probe]` |
| **4. `repr((logits, cache))` (unpacked cell tuple)** | *Pending* | *Pending* | *Pending* (expected `1`) | *Pending* | `[UNFILLED — run probe]` |
| **5. `repr(cache.cache_dict[:8])` (sampled raw dict)** | *Pending* | *Pending* | *Pending* (expected `8`) | *Pending* | `[UNFILLED — note extrapolated full-dict ms & MB]` |
| **6. IDE variable-inspector walk (`cache.items()`)** | *Pending* | *Pending* | *Pending* (expected `0`) | *Pending* | `[UNFILLED — run probe]` |

### 3.3 Live Model Capture vs. Display Split

| Mode & Flags | Cached Keys | Footprint (MiB) | Step 0: `run_with_cache` (ms) | Step 1: `repr(cache)` (ms) | Step 4: `repr((logits, cache))` (ms) | Extrapolated Full `cache_dict` `repr` (ms) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `bridge` (`distilgpt2`, default) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | `[UNFILLED]` |
| `bridge` (`distilgpt2`, `--compat-mode`) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | `[UNFILLED]` |
| `bridge` (`distilgpt2`, `--cache-device cpu`) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | `[UNFILLED]` |
| `hooked-transformer` (`distilgpt2`) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | `[UNFILLED]` |
| **Large-model spot check** (`[UNFILLED model]`) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | `[UNFILLED]` |

### 3.4 Post-Measurement Decision Matrix

| Empirical Outcome | Action to Take |
| :--- | :--- |
| **Python & notebook rendering of `cache` stays fast, and reporter was hitting `(logits, cache)`, `cache.cache_dict`, or `return_cache_object=False`** | Keep `ActivationCache.__repr__` untruncated so hook discovery via `print(cache)` keeps working; document the `logits, cache = model.run_with_cache(...)` unpacking / `return_cache_object=True` pitfall and close or re-scope `#1126`. |
| **Python `repr(cache)` is <5 ms, but a specific notebook frontend stalls on a single 80 KB line at 2,000+ keys** | Implement `_repr_pretty_` (line-wrapped key formatting for IPython) or a high key-count summary threshold (e.g., >200 keys) with an explicit hint like `list(cache.keys())`, mirrored across `HookedTransformer` and `TransformerBridge` workflows with peer `+1`. |
| **Hang only occurs when a notebook variable explorer is open** | Document the IDE variable-inspector interaction (walking `cache.items()` / `cache.model`); evaluate whether `_ipython_display_` or inspector hints mitigate it without breaking mapping semantics. |

## 4. Proposed Maintainer Response for GitHub Issue #1126

Apply label: `needs-repro`

> Hi @repr-rob — thanks for flagging this!
>
> I traced `ActivationCache.__repr__` on `main` (`transformer_lens/ActivationCache.py`), and right now it only materializes the string key list (`f"ActivationCache with keys {list(self.cache_dict.keys())}"`) without formatting or reading any cached tensors. Because many workflows rely on `print(cache)` to see available hook names, we want to make sure we pinpoint what is actually hanging your notebook before we truncate `repr(cache)`.
>
> In practice, a freeze around `run_with_cache` usually comes from one of four adjacent paths rather than `ActivationCache.__repr__` itself:
> 1. **Unpacked return tuple** — running `model.run_with_cache(...)` as the last expression in a cell formats `(logits, cache)`, which runs `torch.Tensor.__repr__` (and a GPU-to-CPU sync) on `logits`.
> 2. **Raw `dict` instead of `ActivationCache`** — passing `return_cache_object=False`, calling `HookedRootModule.run_with_cache` directly, or printing `cache.cache_dict` / `dict(cache)` formats every tensor in the cache.
> 3. **Async GPU capture flushing at print time** — if `run_with_cache` ran on CUDA/MPS, pending kernels (or per-hook `device="cpu"` copies) may only synchronize when the cell goes to print.
> 4. **Notebook UI / variable explorer** — VS Code, JupyterLab, or Colab's variable inspector walking `cache.items()`, or the browser choking on a single very long line of key names.
>
> Could you share a minimal snippet showing how you're creating and printing the cache, which model and API (`TransformerBridge` vs. `HookedTransformer`) you're using, and your notebook frontend (VS Code, JupyterLab, Colab)?
>
> If you still have the notebook handy, running this quick timing split on your existing `cache` object will tell us right away whether Python `repr(cache)` or the frontend/capture step is the culprit:
>
> ```python
> import time, torch
> if torch.cuda.is_available():
>     torch.cuda.synchronize()
> t0 = time.perf_counter()
> s = repr(cache)
> dt_ms = (time.perf_counter() - t0) * 1000
> print(f"type={type(cache).__name__}, keys={len(cache)}, chars={len(s):,}, repr_ms={dt_ms:.2f}")
> ```
>
> Once we see which step is stalling for you, we can either fix the underlying bottleneck or add clean pretty-printing/truncation in the right place!
