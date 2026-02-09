class Packet:
    """
    Represents a data packet with QoS requirements
    """
    
    # Class variable for unique packet ID generation
    _next_id = 0
    
    def __init__(self, user_id, slice_name, arrival_time, size, deadline):
        """
        Initialize a packet
        
        Args:
            user_id (int): ID of the user who generated this packet
            slice_name (str): Name of the slice ('VoLTE', 'Video', 'URLLC')
            arrival_time (int): Arrival time in slots
            size (float): Packet size in bits
            deadline (int): Deadline in slots (relative to arrival)
        """
        self.packet_id = Packet._next_id
        Packet._next_id += 1
        
        self.user_id = user_id
        self.slice_name = slice_name
        self.arrival_time = arrival_time
        self.size = size  # bits
        self.original_size = size  # Keep track of original size
        self.deadline = arrival_time + deadline  # Absolute deadline
        
        # Transmission tracking
        self.transmission_start = None  # When transmission started
        self.transmission_complete = None  # When transmission completed
        self.remaining_size = size  # Bits remaining to transmit
        
    def start_transmission(self, slot):
        """Mark the start of transmission"""
        if self.transmission_start is None:
            self.transmission_start = slot
    
    def transmit(self, bits_transmitted):
        """
        Transmit some bits of this packet
        
        Args:
            bits_transmitted (float): Number of bits transmitted
            
        Returns:
            bool: True if packet is fully transmitted
        """
        self.remaining_size -= bits_transmitted
        if self.remaining_size <= 0:
            self.remaining_size = 0
            return True  # Fully transmitted
        return False  # Still has remaining bits
    
    def complete_transmission(self, slot):
        """Mark the completion of transmission"""
        self.transmission_complete = slot
    
    def get_latency(self):
        """
        Calculate total latency (queueing + transmission)
        
        Returns:
            int: Latency in slots, or None if not completed
        """
        if self.transmission_complete is not None:
            return self.transmission_complete - self.arrival_time
        return None
    
    def is_deadline_met(self):
        """
        Check if packet met its deadline
        
        Returns:
            bool: True if deadline met, False otherwise
        """
        if self.transmission_complete is None:
            return False  # Not yet transmitted
        return self.transmission_complete <= self.deadline
    
    def __repr__(self):
        return (f"Packet(id={self.packet_id}, user={self.user_id}, "
                f"slice={self.slice_name}, arrival={self.arrival_time}, "
                f"size={self.original_size:.0f}bits, deadline={self.deadline})")