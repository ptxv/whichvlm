import json
from argparse import Namespace
from pathlib import Path


def test_runtime_smoke_runs_generated_backend_and_records_result(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from benchmarks import real_hardware

    image = tmp_path / "image.jpg"
    output = tmp_path / "result.json"
    args = Namespace(
        backend="llama.cpp",
        model="org/model-gguf",
        architecture="qwen2_vl",
        model_file="model-q4_k_m.gguf",
        projector_file="mmproj-model-f16.gguf",
        image=image,
        context=2048,
        max_tokens=4,
        cpu_only=True,
        output=output,
    )
    generated = {}
    executed = {}
    hardware = object()

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
    monkeypatch.setattr("hardware.detector.detect_hardware", lambda: hardware)
    monkeypatch.setattr(real_hardware.subprocess, "run", fake_run)
    monkeypatch.setattr(real_hardware.metadata, "version", lambda _: "1.2.3")

    real_hardware.runtime_smoke(args)

    assert generated["variant"].filename == "model-q4_k_m.gguf"
    assert generated["model"].model_format == "gguf"
    assert generated["model"].artifacts[0].filename == "mmproj-model-f16.gguf"
    assert generated["cpu_only"] is True
    assert generated["kwargs"]["backend_name"] == "llama.cpp"
    assert generated["kwargs"]["hardware"] is hardware
    assert executed["command"] == [real_hardware.sys.executable, "-c", "print('smoke')"]
    assert executed["kwargs"] == {
        "input": "Describe this image in one sentence.\nexit\n",
        "text": True,
        "check": True,
    }
    assert json.loads(output.read_text()) == {
        "backend": "llama.cpp",
        "version": "1.2.3",
        "model": "org/model-gguf",
    }
