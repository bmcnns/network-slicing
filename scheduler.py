# scheduler.py
from config import Config

class RoundRobinScheduler:
    """
    Round Robin scheduler for network slicing
    Based on 3GPP TR 36.814 scheduling principles
    
    Implements per-user FIFO queues with round-robin service across users
    within a slice to ensure fairness.
    """
    
    def __init__(self, slice_name):
        """
        Initialize scheduler
        
        Args:
            slice_name (str): Name of the slice ('VoLTE', 'Video', 'URLLC')
        """
        self.slice_name = slice_name
        self.user_queues = {}  # Maps user_id to list of packets (FIFO per user)
        self.current_packet = None  # Packet currently being transmitted
        self.user_order = []  # List of user IDs for round robin scheduling
        self.current_user_index = 0  # Index for round-robin user selection
        self.last_served_user = None  # Track last served user for fairness
    
    def set_users(self, user_ids):
        """
        Initialize per-user queues for all users in this slice
        
        Args:
            user_ids (list): List of user IDs belonging to this slice
        """
        self.user_order = list(user_ids)
        self.current_user_index = 0
        
        # Create FIFO queue for each user
        for user_id in user_ids:
            self.user_queues[user_id] = []
    
    def enqueue(self, packet):
        """
        Enqueue packet to corresponding user's queue
        
        Args:
            packet (Packet): Packet to enqueue with user_id attribute
            
        Returns:
            bool: True if successfully enqueued, False if dropped due to queue overflow
        """
        user_id = packet.user_id
        
        # Check if user's queue has reached maximum capacity
        if len(self.user_queues[user_id]) >= Config.MAX_QUEUE_SIZE:
            return False  # Packet dropped
        
        self.user_queues[user_id].append(packet)
        return True
    
    def get_next_packet(self):
        """
        Select next packet for transmission using round-robin scheduling
        
        Implements round-robin across users to ensure fairness.
        Continues transmitting current packet if not yet complete.
        
        Returns:
            Packet or None: Next packet to transmit, or None if all queues empty
        """
        # Continue transmitting current packet if incomplete
        if self.current_packet is not None:
            return self.current_packet
        
        # No users in this slice
        if len(self.user_order) == 0:
            return None
        
        # Round-robin: find next user with packets
        attempts = 0
        while attempts < len(self.user_order):
            user_id = self.user_order[self.current_user_index]
            
            # Move to next user for next call (round-robin advancement)
            self.current_user_index = (self.current_user_index + 1) % len(self.user_order)
            attempts += 1
            
            # Select packet from this user if queue non-empty
            if len(self.user_queues[user_id]) > 0:
                self.current_packet = self.user_queues[user_id][0]
                self.last_served_user = user_id
                return self.current_packet
        
        # All user queues are empty
        return None
    
    def transmit(self, bits_transmitted):
        """
        Transmit bits from current packet
        
        Follows packet continuation strategy: if packet not completed
        in one slot, continue in next slot (Method A in system model).
        
        Args:
            bits_transmitted (float): Number of bits transmitted in this slot
            
        Returns:
            Packet or None: Completed packet if fully transmitted, None otherwise
        """
        if self.current_packet is None:
            return None
        
        # Update packet's remaining bits
        is_complete = self.current_packet.transmit(bits_transmitted)
        
        if is_complete:
            # Packet fully transmitted
            user_id = self.current_packet.user_id
            completed_packet = self.current_packet
            
            # Remove completed packet from user's queue
            if completed_packet in self.user_queues[user_id]:
                self.user_queues[user_id].remove(completed_packet)
            
            self.current_packet = None
            return completed_packet
        
        # Packet incomplete, will continue in next slot
        return None
    
    def get_queue_length(self):
        """
        Get total queue length across all users
        
        Returns:
            int: Total number of packets queued in this slice
        """
        return sum(len(queue) for queue in self.user_queues.values())
    
    def get_user_queue_length(self, user_id):
        """
        Get queue length for specific user
        
        Args:
            user_id (int): User ID
            
        Returns:
            int: Number of packets in user's queue
        """
        return len(self.user_queues.get(user_id, []))
    
    @property
    def queue(self):
        """
        Backward compatibility: merge all user queues
        
        Returns:
            list: All packets across all users (read-only view)
        """
        all_packets = []
        for user_id in self.user_order:
            all_packets.extend(self.user_queues[user_id])
        return all_packets
    
    def clear_all_queues(self):
        """
        Clear all user queues and reset transmission state
        
        Used at the start of each decision step to ensure clean state.
        """
        for user_id in self.user_order:
            self.user_queues[user_id].clear()
        self.current_packet = None