import json
import os
import shutil
import subprocess
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def write_e2e_cache(cache_home: Path) -> None:
    cache_dir = cache_home / "whichvlm"
    cache_dir.mkdir(parents=True)
    cached_at = time.time()
    model = {
        "id": MODEL_ID,
        "family_id": "qwen-vl",
        "name": "Qwen2.5-VL-7B-Instruct",
        "parameter_count": 7_000_000_000,
        "downloads": 1000,
        "likes": 100,
        "gguf_variants": [
            {
                "filename": "qwen2.5-vl-7b-q4_k_m.gguf",
                "quant_type": "Q4_K_M",
                "file_size_bytes": 4_000_000_000,
            }
        ],
        "hf_pipeline_tag": "image-text-to-text",
        "tags": ["vision", "image-text-to-text"],
        "model_format": "safetensors",
    }
    (cache_dir / "models.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cached_at": cached_at,
                "ttl_seconds": 6 * 3600,
                "source": {"name": "e2e-fixture"},
                "models": [model],
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cached_at": cached_at,
                "ttl_seconds": 24 * 3600,
                "source": {"name": "e2e-fixture"},
                "scores": {MODEL_ID: 72.5},
            }
        ),
        encoding="utf-8",
    )


def e2e_env(tmp_path: Path) -> dict[str, str]:
    cache_home = tmp_path / "cache"
    write_e2e_cache(cache_home)
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache_home)
    env["TERM"] = "dumb"
    return env


def run_whichvlm(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    exe = shutil.which("whichvlm")
    assert exe is not None
    return subprocess.run(
        [exe, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
        check=False,
    )


def assert_success(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def json_stdout(stdout: str) -> dict:
    start = stdout.find("{")
    assert start >= 0, stdout
    data, _ = json.JSONDecoder().raw_decode(stdout[start:])
    return data


def test_installed_cli_help() -> None:
    result = run_whichvlm(["--help"], os.environ.copy())

    assert_success(result)
    assert "Find local vision-language models" in result.stdout
    assert "hardware-plan" in result.stdout


def test_installed_cli_hardware_with_simulated_gpu(tmp_path: Path) -> None:
    result = run_whichvlm(["hardware", "--gpu", "RTX 4090"], e2e_env(tmp_path))

    assert_success(result)
    assert "GPU 0" in result.stdout
    assert "RTX 4090" in result.stdout


def test_installed_cli_hardware_plan_json_from_cache(tmp_path: Path) -> None:
    result = run_whichvlm(
        ["hardware-plan", "RTX 4090", "--top", "1", "--json"],
        e2e_env(tmp_path),
    )

    assert_success(result)
    data = json_stdout(result.stdout)
    assert "RTX 4090" in data["hardware"]["gpus"][0]["name"]
    assert data["models"][0]["model_id"] == MODEL_ID


def test_installed_cli_plan_json_from_cache(tmp_path: Path) -> None:
    result = run_whichvlm(["plan", MODEL_ID, "--json"], e2e_env(tmp_path))

    assert_success(result)
    data = json_stdout(result.stdout)
    assert data["model"]["id"] == MODEL_ID
    assert "vram_by_quant" in data
    assert "reverse_lookup" in data


def test_installed_cli_snippet_generates_script(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fixture")

    result = run_whichvlm(
        ["snippet", MODEL_ID, "--image", str(image_path), "--backend", "transformers"],
        e2e_env(tmp_path),
    )

    assert_success(result)
    assert MODEL_ID in result.stdout
    assert "AutoProcessor" in result.stdout
    assert "image_path =" in result.stdout
    assert "Image.open(image_path)" in result.stdout
