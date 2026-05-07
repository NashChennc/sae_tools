import importlib.util
import sys
from pathlib import Path


def _load_runner():
    path = Path("scripts/run_idle_gpu_workflow.py")
    spec = importlib.util.spec_from_file_location("run_idle_gpu_workflow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_idle_gpu_filter_requires_zero_util_and_free_memory():
    runner = _load_runner()
    gpus = [
        runner.GPUInfo(index=0, name="A40", memory_total_mib=46068, memory_used_mib=7, utilization_gpu_pct=0),
        runner.GPUInfo(index=3, name="A40", memory_total_mib=46068, memory_used_mib=29458, utilization_gpu_pct=82),
        runner.GPUInfo(index=4, name="A40", memory_total_mib=46068, memory_used_mib=3146, utilization_gpu_pct=0),
        runner.GPUInfo(index=6, name="A40", memory_total_mib=46068, memory_used_mib=42712, utilization_gpu_pct=0),
    ]

    selected, rows = runner.classify_gpus(
        gpus,
        max_gpu_util=0,
        max_used_mib=512,
        min_free_mib=30000,
        include=None,
        exclude=None,
    )

    assert [gpu.index for gpu in selected] == [0]
    reasons = {row["gpu"]: row["reason"] for row in rows}
    assert "utilization" in reasons[3]
    assert "used memory" in reasons[4]
    assert "free memory" in reasons[6]
