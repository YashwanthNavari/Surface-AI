"""
Computational Efficiency Profiler
Measures model file size, parameter counts, forward inference latency (ms/image), and throughput (FPS).
"""
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from src.config import MODELS_DIR

def profile_keras_model(model_path, input_shape=(1, 224, 224, 3), num_warmup=10, num_runs=50):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    model = tf.keras.models.load_model(str(path))

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

    # Warm-up inference runs
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    for _ in range(num_warmup):
        _ = model(dummy_input, training=False)

    # Timed inference benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_input, training=False)
        latencies.append(time.perf_counter() - start)

    avg_latency_ms = float(np.mean(latencies) * 1000)
    std_latency_ms = float(np.std(latencies) * 1000)
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "model_file": path.name,
        "disk_size_mb": round(size_mb, 2),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "latency_ms_mean": round(avg_latency_ms, 2),
        "latency_ms_std": round(std_latency_ms, 2),
        "throughput_fps": round(fps, 1),
    }

if __name__ == "__main__":
    for m in ["custom_cnn.keras", "efficientnet_frozen.keras", "efficientnet_finetuned.keras"]:
        p = MODELS_DIR / m
        if p.exists():
            print(profile_keras_model(p))
