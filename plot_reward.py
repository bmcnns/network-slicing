import matplotlib.pyplot as plt
import numpy as np

steps = []
rewards = []

with open("training_ppo.dat", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Step"):
            continue
        
        parts = line.split()
        step = int(parts[0])
        reward = float(parts[3])

        steps.append(step)
        rewards.append(reward)

steps = np.array(steps)
rewards = np.array(rewards)

window = 100 
if len(rewards) >= window:
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    smoothed_steps = steps[window-1:]
else:
    smoothed = rewards
    smoothed_steps = steps

plt.figure(figsize=(7, 5))
plt.plot(steps, rewards, alpha=0.3, label="Raw reward")
plt.plot(smoothed_steps, smoothed, linewidth=2, label="Smoothed reward")

plt.xlabel("Round")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
