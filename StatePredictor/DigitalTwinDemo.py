import numpy as np
from DigitalTwin.DigitalTwin import DigitalTwin

T = 200
D = 3

# build a simple dataset for testing
t = np.arange(T, dtype=np.float32)
states = np.stack([
    np.sin(0.1 * t),
    np.cos(0.1 * t),
    np.sin(0.05 * t),
], axis=1).astype(np.float32)

rewards = np.zeros((T,), dtype=np.float32)

env = DigitalTwin(
    model_path="./model_components/test_dt.pt",
    scaler_path="./model_components/test_standardizer.json",
    run_config_path="./model_components/test_run_config.json",
    log_path="./log/test_train_log.jsonl",
    window_size=8,
    epochs=5,
    batch_size=16,
    device="cpu",
)

info = env.retrain(states, rewards)
print(info)
print("model loaded:", env.model is not None)
print("scaler loaded:", env.scaler is not None)