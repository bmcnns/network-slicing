from Simulator.config import Config

class RoundRobinScheduler:
    """
    Round Robin scheduler for a slice
    If packet not finished, continue in next slot
    """
    
    def __init__(self, slice_name):
        """
        Initialize scheduler
        
        Args:
            slice_name (str): Name of the slice
        """
        self.slice_name = slice_name
        self.queue = []  # FIFO queue of packets
        self.current_packet = None  # Packet currently being transmitted
        self.user_order = []  # List of user IDs for round robin
        self.current_user_index = 0
    
    def set_users(self, user_ids):
        """
        Set the list of users in this slice
        
        Args:
            user_ids (list): List of user IDs
        """
        self.user_order = list(user_ids)
        self.current_user_index = 0
    
    def enqueue(self, packet):
        """
        Add packet to queue
        
        Args:
            packet (Packet): Packet to enqueue
            
        Returns:
            bool: True if enqueued, False if queue full (dropped)
        """
        if len(self.queue) >= Config.MAX_QUEUE_SIZE:
            # Queue full, drop packet
            return False
        
        self.queue.append(packet)
        return True
    
    def get_next_packet(self):
        """
        Get next packet to transmit according to Round Robin
        
        Returns:
            Packet or None: Next packet to transmit
        """
        # If currently transmitting a packet, continue with it
        if self.current_packet is not None:
            return self.current_packet
        
        # Otherwise, get next packet from queue
        if len(self.queue) == 0:
            return None
        
        # Round Robin: find next packet from next user in rotation
        if len(self.user_order) == 0:
            return None
        
        # Simple FIFO for now (head of queue)
        # In true round-robin per-user, would track which user's turn
        self.current_packet = self.queue[0]
        return self.current_packet
    
    def transmit(self, bits_transmitted):
        """
        Transmit bits from current packet
        
        Args:
            bits_transmitted (float): Bits transmitted this slot
            
        Returns:
            Packet or None: Completed packet if transmission finished
        """
        if self.current_packet is None:
            return None
        
        # Transmit bits
        is_complete = self.current_packet.transmit(bits_transmitted)
        
        if is_complete:
            # Packet fully transmitted
            completed_packet = self.current_packet
            self.queue.remove(self.current_packet)
            self.current_packet = None
            return completed_packet
        
        # Packet not yet complete, will continue next slot
        return None
    
    def get_queue_length(self):
        """Get current queue length"""
        return len(self.queue)