from scheduler import RoundRobinScheduler
from traffic import TrafficGenerator
from config import Config
import numpy as np

class NetworkSlice:
    """
    Represents a network slice (VoLTE, Video, or URLLC)
    """
    
    def __init__(self, slice_name, user_ids, channel_model, random_state=None, env=None):
        """
        Initialize network slice
        
        Args:
            slice_name (str): Name of slice
            user_ids (list): List of user IDs in this slice
            channel_model (ChannelModel): Channel model instance
            random_state (np.random.RandomState): Random state
        """
        self.slice_name = slice_name
        self.user_ids = user_ids
        self.channel_model = channel_model
        self.random_state = random_state if random_state else np.random.RandomState()
        
        # Create scheduler
        self.scheduler = RoundRobinScheduler(slice_name)
        self.scheduler.set_users(user_ids)
        
        # Create traffic generators for each user
        self.traffic_generators = {}
        for user_id in user_ids:
            self.traffic_generators[user_id] = TrafficGenerator(
                user_id, slice_name, self.random_state
            )
        
        # Allocated resources
        self.bandwidth = 0  # Will be set by environment
        self.power = 0  # Will be set by environment
        
        # Statistics
        self.total_arrived_packets = 0
        self.total_successful_packets = 0  # x_n_k = 1
        self.dropped_packets = 0
    
    def generate_traffic(self, duration_slots, start_slot=0):
        """
        Generate traffic for all users in this slice
        
        Args:
            duration_slots (int): Duration in slots
            start_slot (int): Global slot number for start of this period
            
        Returns:
            list of Packet: All generated packets
        """
        all_packets = []
        for user_id in self.user_ids:
            packets = self.traffic_generators[user_id].generate_arrivals(
                duration_slots,
                start_slot=start_slot
                )
            all_packets.extend(packets)
        
        # Sort by arrival time
        all_packets.sort(key=lambda p: p.arrival_time)

        return all_packets
    
    def enqueue_packet(self, packet):
        """
        Enqueue a packet
        
        Args:
            packet (Packet): Packet to enqueue
            
        Returns:
            bool: True if enqueued, False if dropped
        """
        self.total_arrived_packets += 1
        success = self.scheduler.enqueue(packet)
        if not success:
            self.dropped_packets += 1
        return success
    
    def get_ssr(self):
        """
        Calculate Slice Success Rate (SSR)
        
        Returns:
            float: SSR value between 0 and 1
        """
        if self.total_arrived_packets == 0:
            return 1.0  # No packets arrived, trivially satisfied
        
        return self.total_successful_packets / self.total_arrived_packets
    
    def reset_statistics(self):
        """Reset statistics for new step"""
        self.total_arrived_packets = 0
        self.total_successful_packets = 0
        self.dropped_packets = 0