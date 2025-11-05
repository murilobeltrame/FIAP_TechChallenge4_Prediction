import psutil
import torch

def get_system_metrics():
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory_percent": psutil.virtual_memory().percent,
    }

    if torch.cuda.is_available():
        metrics["gpu_name"] = torch.cuda.get_device_name(0)
        metrics["gpu_memory_allocated_MB"] = round(torch.cuda.memory_allocated(0) / 1024**2, 2)
        metrics["gpu_memory_total_MB"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2)
    else:
        metrics["gpu_name"] = None
    return metrics

