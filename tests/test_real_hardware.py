import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "real_hardware", Path(__file__).parents[1] / "benchmarks" / "real_hardware.py"
)
assert spec is not None
assert spec.loader is not None
real_hardware = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_hardware)


def test_runtime_smoke_runs_generated_backend_and_records_result(monkeypatch, tmp_path):
    image = tmp_path / "image.jpg"
    output = tmp_path / "result.json"
    args = real_hardware.parser().parse_args(
        [
            "runtime-smoke",
            "--backend",
            "llama.cpp",
            "--model",
            "org/model-gguf",
            "--architecture",
            "qwen2_vl",
            "--model-file",
            "model-q4_k_m.gguf",
            "--projector-file",
            "mmproj-model-f16.gguf",
            "--image",
            str(image),
            "--cpu-only",
            "--output",
            str(output),
        ]
    )
    generated = {}
    executed = {}

    def fake_generate(model, variant, context, cpu_only, **kwargs):
        generated.update(
            model=model,
            variant=variant,
            context=context,
            cpu_only=cpu_only,
            kwargs=kwargs,
        )
        return "print('smoke')"

    def fake_run(command, **kwargs):
        executed.update(command=command, kwargs=kwargs)

    monkeypatch.setattr("runtime.generate_run_script", fake_generate)
    monkeypatch.setattr(real_hardware.subprocess, "run", fake_run)
    monkeypatch.setattr(real_hardware.metadata, "version", lambda _: "1.2.3")

    args.func(args)

    assert generated["variant"].filename == "model-q4_k_m.gguf"
    assert generated["model"].model_format == "gguf"
    assert generated["model"].artifacts[0].filename == "mmproj-model-f16.gguf"
    assert generated["cpu_only"] is True
    assert generated["kwargs"]["backend_name"] == "llama.cpp"
    assert executed["command"] == [real_hardware.sys.executable, "-c", "print('smoke')"]
    assert executed["kwargs"]["check"] is True
    assert json.loads(output.read_text()) == {
        "backend": "llama.cpp",
        "version": "1.2.3",
        "model": "org/model-gguf",
        "status": "passed",
    }
