import numpy as np
from Simulator.environment import NetworkSlicingEnv
from Simulator.metrics import MetricsTracker
from Simulator.config import Config

def test_random_policy(n_steps=10):
    """
    Test environment with random bandwidth allocation policy
    
    Args:
        n_steps (int): Number of steps to run
    """
    print("="*60)
    print("Testing Network Slicing Environment")
    print("="*60)
    
    # Create environment
    env = NetworkSlicingEnv(seed=Config.RANDOM_SEED)
    
    # Create metrics tracker
    tracker = MetricsTracker()
        
    # Run simulation
    print(f"\nRunning {n_steps} steps with random policy...")
    
    for step in range(n_steps):
        # Random action: random bandwidth allocation
        # Ensure sum = 20 MHz
        action = np.random.dirichlet([1, 1, 1]) * 20  # MHz
        
        # print(f"Step {step}, Atcion Before setting: {action}\n")

        # Execute step
        state, reward, done, info = env.step(action)
        
        # Update metrics
        tracker.update(info, reward)
        
        # Print step info
        print(f"\nStep {step + 1}:")
        print(f"  Action (MHz): {action}")
        print(f"  Spectrum Efficiency: {info['spectrum_efficiency']:.4f}")
        print(f"  SSR: {info['ssr']}")
        print(f"  Utility: {info['utility']:.4f}")
        print(f"  Reward: {reward:.4f}")
        print(f"  State: {state}")        
    
    # Print summary
    # tracker.print_summary()




if __name__ == "__main__":
    # Test 1: Random policy
    test_random_policy(n_steps=5)
    
