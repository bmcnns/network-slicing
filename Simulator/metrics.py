import numpy as np
from Simulator.config import Config

class MetricsTracker:
    """
    Track and calculate performance metrics over multiple steps
    """
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.history = {
            'spectrum_efficiency': [],
            'ssr': {slice_name: [] for slice_name in Config.SLICE_NAMES},
            'utility': [],
            'reward': [],
            'bandwidth_allocation': {slice_name: [] for slice_name in Config.SLICE_NAMES},
            'arrived_packets': {slice_name: [] for slice_name in Config.SLICE_NAMES},
            'successful_packets': {slice_name: [] for slice_name in Config.SLICE_NAMES},
            'dropped_packets': {slice_name: [] for slice_name in Config.SLICE_NAMES}
        }

    
    def update(self, metrics, reward):
        """
        Update metrics history
        
        Args:
            metrics (dict): Metrics from environment step
            reward (float): Reward value
        """
        self.history['spectrum_efficiency'].append(metrics['spectrum_efficiency'])
        self.history['utility'].append(metrics['utility'])
        self.history['reward'].append(reward)
        
        for slice_name in Config.SLICE_NAMES:
            self.history['ssr'][slice_name].append(metrics['ssr'][slice_name])
            
            bw = metrics['bandwidth_allocation'][slice_name] / 1e6  # Convert to MHz
            self.history['bandwidth_allocation'][slice_name].append(bw)
            
            self.history['arrived_packets'][slice_name].append(
                metrics['total_arrived_packets'][slice_name]
            )
            self.history['successful_packets'][slice_name].append(
                metrics['total_successful_packets'][slice_name]
            )
            self.history['dropped_packets'][slice_name].append(
                metrics['dropped_packets'][slice_name]
            )
    
    def get_summary(self):
        """
        Get summary statistics
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'avg_spectrum_efficiency': np.mean(self.history['spectrum_efficiency']),
            'avg_utility': np.mean(self.history['utility']),
            'avg_reward': np.mean(self.history['reward']),
            'avg_ssr': {},
            'avg_bandwidth': {},
            'total_arrived': {},
            'total_successful': {},
            'total_dropped': {}
        }
        
        for slice_name in Config.SLICE_NAMES:
            summary['avg_ssr'][slice_name] = np.mean(self.history['ssr'][slice_name])
            summary['avg_bandwidth'][slice_name] = np.mean(
                self.history['bandwidth_allocation'][slice_name]
            )
            summary['total_arrived'][slice_name] = np.sum(
                self.history['arrived_packets'][slice_name]
            )
            summary['total_successful'][slice_name] = np.sum(
                self.history['successful_packets'][slice_name]
            )
            summary['total_dropped'][slice_name] = np.sum(
                self.history['dropped_packets'][slice_name]
            )
        
        return summary
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Average Spectrum Efficiency: {summary['avg_spectrum_efficiency']:.4f}")
        print(f"  Average Utility: {summary['avg_utility']:.4f}")
        print(f"  Average Reward: {summary['avg_reward']:.4f}")
        
        print(f"\nPer-Slice Metrics:")
        for slice_name in Config.SLICE_NAMES:
            print(f"\n  {slice_name}:")
            print(f"    Average SSR: {summary['avg_ssr'][slice_name]:.4f}")
            print(f"    Average Bandwidth: {summary['avg_bandwidth'][slice_name]:.2f} MHz")
            print(f"    Total Arrived: {summary['total_arrived'][slice_name]}")
            print(f"    Total Successful: {summary['total_successful'][slice_name]}")
            print(f"    Total Dropped: {summary['total_dropped'][slice_name]}")
            
            if summary['total_arrived'][slice_name] > 0:
                success_rate = (summary['total_successful'][slice_name] / 
                              summary['total_arrived'][slice_name])
                print(f"    Overall Success Rate: {success_rate:.4f}")
        
        print("\n" + "="*60)