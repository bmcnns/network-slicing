# scheduler.py
from Simulator.config import Config

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
        
        self.dropped_due_to_timeout = 0 
        self.dropped_due_to_overflow = 0

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
            self.dropped_due_to_overflow += 1
            return False  # Packet dropped
        
        self.user_queues[user_id].append(packet)
        return True
    
    def check_and_drop_expired(self, current_slot):
        """
        Check all queues and drop packets that have exceeded their deadline.
                
        Args:
            current_slot (int): Current simulation slot number
            
        Returns:
            list: List of dropped Packet objects (for statistics/logging)
        """
        dropped_packets = []
        
        for user_id in self.user_order:
            queue = self.user_queues[user_id]
            valid_packets = []
            
            for packet in queue:
                # Calculate how long packet has exceeded its deadline
                slots_overdue = current_slot - packet.deadline
                
                # Drop only if significantly overdue
                if slots_overdue > Config.DEADLINE_TOLERANCE:
                    # Deadline violated beyond tolerance - drop packet
                    dropped_packets.append(packet)
                    self.dropped_due_to_timeout += 1
                else:
                    # Still valid or within tolerance - keep in queue
                    valid_packets.append(packet)
            
            # Update queue with only valid packets
            self.user_queues[user_id] = valid_packets
        
        return dropped_packets


    def get_next_packet(self, current_slot):
        """
        Select next packet for transmission using round-robin scheduling
        
        Implements round-robin across users to ensure fairness.
        Continues transmitting current packet if not yet complete.
        
        Returns:
            Packet or None: Next packet to transmit, or None if all queues empty
        """
        dropped_in_this_call = []

        # Check current packet
        if self.current_packet is not None:
            slots_overdue = current_slot - self.current_packet.deadline

            if slots_overdue > Config.DEADLINE_TOLERANCE:
                # Current packet exceeded deadline beyond tolerance - drop it
                user_id = self.current_packet.user_id
                if self.current_packet in self.user_queues[user_id]:
                    self.user_queues[user_id].remove(self.current_packet)

                dropped_in_this_call.append(self.current_packet)
                self.dropped_due_to_timeout += 1
                self.current_packet = None
                # Fall through to select next packet
            else:
                # Current packet still valid or within tolerance - continue transmission
                return self.current_packet, dropped_in_this_call

        # No users in this slice
        if len(self.user_order) == 0:
            return None, dropped_in_this_call
        
        # Round-robin: find next user with packets
        attempts = 0
        while attempts < len(self.user_order):
            user_id = self.user_order[self.current_user_index]
            
            # Move to next user for next call (round-robin advancement)
            self.current_user_index = (self.current_user_index + 1) % len(self.user_order)
            attempts += 1
            
            # Select packet from this user if queue non-empty
            while len(self.user_queues[user_id]) > 0:
                candidate = self.user_queues[user_id][0]
            
                # Calculate how overdue the candidate packet is
                slots_overdue = current_slot - candidate.deadline
            
                # Verify candidate packet hasn't exceeded tolerance
                if slots_overdue > Config.DEADLINE_TOLERANCE:
                    # Expired beyond tolerance - drop and continue searching
                    self.user_queues[user_id].pop(0)
                    dropped_in_this_call.append(candidate)
                    self.dropped_due_to_timeout += 1
                    # Continue inner while loop to check next packet from same user
                else:
                    self.current_packet = self.user_queues[user_id][0]
                    self.last_served_user = user_id
                    return self.current_packet, dropped_in_this_call
        
        # All user queues are empty
        return None, dropped_in_this_call
    
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

        # Reset drop statistics
        self.dropped_due_to_timeout = 0
        self.dropped_due_to_overflow = 0


    def get_drop_statistics(self):
        """
        Get detailed statistics about dropped packets.
        
        Returns:
            dict: Dictionary containing:
                - 'timeout' (int): Packets dropped due to deadline violation
                - 'overflow' (int): Packets dropped due to queue overflow
                - 'total' (int): Total packets dropped
        """
        return {
            'timeout': self.dropped_due_to_timeout,
            'overflow': self.dropped_due_to_overflow,
            'total': self.dropped_due_to_timeout + self.dropped_due_to_overflow
        }
