# whichvlm

Find local vision-language models that fit your machine.

`whichvlm` detects GPU, CPU, RAM, Apple Metal/MLX readiness, model formats, quantized variants, GGUF projectors, and model lineage. It ranks VLM candidates for local inference instead of treating every Hugging Face repo as one plain text model.

## Install

Use Python 3.11 or newer.

Run from source without installing:

```bash
uvx --from git+https://github.com/ptxv/whichvlm.git whichvlm --help
```

Or install from source with pip:

```bash
python -m pip install "whichvlm @ git+https://github.com/ptxv/whichvlm.git"
whichvlm --help
```

## Use

The examples below use the installed command. To run without installing, replace
`whichvlm` with `uvx --from git+https://github.com/ptxv/whichvlm.git whichvlm`.

![whichvlm CLI demo](https://raw.githubusercontent.com/ptxv/whichvlm/master/assets/whichvlm-demo.gif)

Rank VLMs directly with `whichvlm`:

```bash
whichvlm
whichvlm list
whichvlm --refresh --profile vision
```

Simulate Apple Silicon or a discrete GPU:

```bash
whichvlm --gpu "Apple M3 Max"
whichvlm --gpu "RTX 4090" --vram-headroom 10%
whichvlm --gpu "RTX 4090" --perf-vram 10%
```

Override incomplete detected GPU metadata:

```bash
whichvlm --vram 12 --memory-bandwidth 288
whichvlm --gpu-index 1 --vram 24 --memory-bandwidth 960
```

Return machine-readable output:

```bash
whichvlm --json --top 5
```

Change the VLM workload estimate:

```bash
whichvlm --image-count 2 --image-size 896 --context-length 8192
```

Reserve memory for backend overhead or performance features:

```bash
whichvlm --gpu "RTX 4090" --vram-headroom auto --perf-vram 10%
whichvlm plan Qwen/Qwen2.5-VL-7B-Instruct --perf-vram 10%
whichvlm hardware-plan "RTX 4090" --perf-vram 10%
whichvlm upgrade "RTX 4090" "RTX 5090" --perf-vram 10%
```

Only show full GPU fits:

```bash
whichvlm --gpu-only
whichvlm --fit full-gpu
```

## Run A Model

VLM runners require an image path.

```bash
whichvlm run Qwen/Qwen2.5-VL-7B-Instruct --image ./image.jpg --max-tokens 256
whichvlm run Qwen/Qwen2.5-VL-7B-Instruct --backend transformers --image ./before.jpg --image ./after.jpg
whichvlm snippet Qwen/Qwen2.5-VL-7B-Instruct --image ./image.jpg --context-length 8192
```

Runtime memory budgets are fractions of each GPU's total memory. `--perf-vram`
derives that fraction from the usable VRAM budget. Transformers receives the
effective byte limit through `max_memory`; vLLM and SGLang receive the fraction
through their native backend options. Runtime commands show both the effective
bytes and translated backend value.

```bash
whichvlm run Qwen/Qwen2.5-VL-7B-Instruct --backend vllm --perf-vram 10% --image ./image.jpg
whichvlm serve Qwen/Qwen2.5-VL-7B-Instruct --backend sglang --gpu-memory-utilization 0.82
```

Runtime support is intentionally guarded:

- Transformers VLMs use `AutoProcessor` and image/text chat templates.
- GGUF VLMs require a concrete GGUF file plus an `mmproj` or projector artifact.
- MLX VLMs require a concrete MLX model package.
- Text-only GGUF and Transformers paths remain available for inherited core behavior.

## What It Models

`whichvlm` tracks a VLM as a package graph:

- `ModelArtifact`: repo, file format, quantization, access, backend support, source kind, filename.
- `ModelComponent`: language tower, vision encoder, projector, processor, tokenizer, merged checkpoint, adapter.
- `ModelLineage`: base models, merged parents, variant relation, and fused/merged status.

The ranker is VLM-aware but conservative. Vision memory includes language weights, KV cache, activation memory, estimated vision encoder/projector overhead, image-token expansion, and prefill scratch. VRAM estimates expose high, medium, or low confidence with an estimated range so calibrated fits and fallback estimates are distinguishable.

## Data Sources

Model metadata comes from Hugging Face API queries, local cache, and curated VLM seeds.
Cache payloads include a SHA-256 checksum of their canonical JSON content and are ignored when the checksum does not match.

The fetcher prioritizes:

- `image-text-to-text`
- `visual-question-answering`
- `image-to-text`
- GGUF, MLX, AWQ, GPTQ, BNB, and FP8 variants
- text-generation only as backbone or variant discovery

Benchmark evidence is graded as direct, variant, base model, interpolated, self-reported, or absent. Vision scores lead the `vision` and `ocr` profiles. Text benchmarks are fallback evidence, and output labels show when ranking evidence is indirect or missing.

## Development

Install the project and test dependencies from a clone:

```bash
uv sync --group dev
```

Run the full suite:

```bash
uv run pytest -q
```

Run focused tests:

```bash
uv run pytest -q tests/test_runtime.py tests/test_fetcher.py tests/test_ranker.py
```

Compile-check source and tests:

```bash
uv run python -m compileall -q src tests
```

The source layout is under `src`. Tests live under `tests`. Avoid importing private CLI helpers in new tests; prefer runtime, ranker, fetcher, or output APIs.

## Real Hardware Benchmarks

These use real hardware, downloads, and runtime dependencies.

Peak-memory calibrations are loaded from
`src/data/vram_calibrations.json`. A record affects estimates only when it
includes the model ID and revision, artifact, GPU, runtime version, benchmark
command, measurement method, and evidence source. The command and source must
be sufficient to reproduce the real-hardware measurement. Evidenced records
may transfer to nearby models in the same architecture when backend, format,
quantization, workload, and MoE state also match.

Run the same detection benchmark on every target machine:

```bash
uv run python benchmarks/real_hardware.py detection --expect-backend metal
```

Run the same GGUF+mmproj VLM benchmark on every target machine:

```bash
uv run python benchmarks/real_hardware.py gguf-mmproj \
  --repo owner/model-gguf \
  --model-file model-q4_k_m.gguf \
  --mmproj-file mmproj-model-f16.gguf \
  --handler Llava16ChatHandler \
  --image ./image.jpg
```

## Current Limits

The model inventory is not complete.

Multimodal ranking and VRAM estimates should be read with their evidence and confidence labels; uncalibrated paths stay marked as estimates.

GGUF VLM and MLX VLM runners are only as reliable as the concrete artifacts and runtime handlers discovered for a model package.

ANE is detected as information only. It is not scored until there is a concrete VLM runtime path.
