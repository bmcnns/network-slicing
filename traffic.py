import numpy as np
from config import Config
from packet import Packet

class TrafficGenerator:
    """
    Generates traffic for a specific user in a specific slice
    """
    
    def __init__(self, user_id, slice_name, random_state=None):
        """
        Initialize traffic generator
        
        Args:
            user_id (int): User ID
            slice_name (str): Slice name ('VoLTE', 'Video', 'URLLC')
            random_state (np.random.RandomState): Random state for reproducibility
        """
        self.user_id = user_id
        self.slice_name = slice_name
        self.random_state = random_state if random_state else np.random.RandomState()
        
        # Get slice-specific parameters
        self.packet_size = Config.get_packet_size(slice_name)
        self.latency_sla = Config.get_latency_sla(slice_name)
    
    def generate_arrivals(self, duration_slots, start_slot=0):
        """
        Generate packet arrivals for a given duration
        
        Args:
            duration_slots (int): Duration in slots (e.g., 400 slots)
            start_slot (int): Global slot number for start (default 0)

        Returns:
            list of Packet: Generated packets
        """
        if self.slice_name == 'VoLTE':
            return self._generate_uniform(duration_slots, start_slot)
        elif self.slice_name == 'Video':
            return self._generate_truncated_exponential(duration_slots, start_slot)
        elif self.slice_name == 'URLLC':
            return self._generate_exponential(duration_slots, start_slot)
        else:
            raise ValueError(f"Unknown slice: {self.slice_name}")

    def _generate_uniform(self, duration_slots, start_slot=0):
        """
        Generate VoLTE traffic with Uniform inter-arrival time
        
        Args:
            duration_slots (int): Duration in slots
            
        Returns:
            list of Packet: Generated packets
        """
        packets = []
        relative_slot = self.random_state.randint(0, Config.SLOTS_PER_STEP)
        
        while relative_slot < duration_slots:
            global_arrival = int(start_slot + relative_slot)

            packet = Packet(
                user_id=self.user_id,
                slice_name=self.slice_name,
                arrival_time=global_arrival,
                size=self.packet_size,
                deadline=self.latency_sla
            )
            packets.append(packet)
            
            # Generate next arrival time
            interval = self.random_state.uniform(
                Config.VOLTE_MIN_INTERVAL,
                Config.VOLTE_MAX_INTERVAL
            )
            relative_slot += interval
        
        return packets
    
    def _generate_truncated_exponential(self, duration_slots, start_slot):
        """
        Generate Video traffic with Truncated Exponential inter-arrival time
        
        Args:
            duration_slots (int): Duration in slots
            
        Returns:
            list of Packet: Generated packets
        """
        packets = []
        relative_slot = self.random_state.randint(0, Config.SLOTS_PER_STEP)
        
        while relative_slot < duration_slots:
            global_arrival = int(start_slot + relative_slot)

            packet = Packet(
                user_id=self.user_id,
                slice_name=self.slice_name,
                arrival_time=global_arrival,
                size=self.packet_size,
                deadline=self.latency_sla
            )
            packets.append(packet)
            
            # Generate next arrival with truncated exponential
            raw_interval = self.random_state.exponential(1.0 / Config.VIDEO_LAMBDA)
            interval = min(raw_interval, Config.VIDEO_MAX_INTERVAL)
            relative_slot += interval
        
        return packets
    
    def _generate_exponential(self, duration_slots, start_slot=0):
        """
        Generate URLLC traffic with Exponential inter-arrival time
        
        Args:
            duration_slots (int): Duration in slots
            
        Returns:
            list of Packet: Generated packets
        """
        packets = []
        
        mean_interval = 1.0 / Config.URLLC_LAMBDA
        relative_slot = self.random_state.uniform(0, mean_interval)

        while relative_slot < duration_slots:
            global_arrival = int(start_slot + relative_slot)
            packet = Packet(
                user_id=self.user_id,
                slice_name=self.slice_name,
                arrival_time=global_arrival,
                size=self.packet_size,
                deadline=self.latency_sla
            )
            packets.append(packet)
            
            # Generate next arrival with exponential
            interval = self.random_state.exponential(1.0 / Config.URLLC_LAMBDA)
            relative_slot += interval
        
        return packets