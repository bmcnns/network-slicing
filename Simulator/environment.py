import numpy as np
from .config import Config
from .channel import ChannelModel
from .network_slice import NetworkSlice
from .packet import Packet

class NetworkSlicingEnv:
    """
    Network slicing simulation environment
    Simulates the complete system for one decision step (400 slots)
    """
    
    def __init__(self, seed=None):
        """
        Initialize environment
        
        Args:
            seed (int): Random seed for reproducibility
        """
        if seed is None:
            seed = Config.RANDOM_SEED
        
        self.random_state = np.random.RandomState(seed)
        np.random.seed(seed)
        
        # Initialize channel model
        self.channel_model = ChannelModel()
        
        # Initialize users
        self._init_users() 
        
        # Initialize slices
        self._init_slices()
        
        # Simulation state
        self.current_step = 0
        self.global_slot = 0

        # Store user rate statistics for rate SLA checking
        self.user_transmitted_bits = {}  # user_id -> total bits transmitted
        self.user_packets_in_step = {}  # user_id -> list of packets
        
    def _init_users(self):
        """Initialize user positions and channel gains"""
        self.n_users = Config.N_USERS
        
        # Randomly distribute users in circular cell
        angles = self.random_state.uniform(0, 2*np.pi, self.n_users)
        # distances = self.random_state.uniform(0, Config.CELL_RADIUS, self.n_users)
        distances = self.random_state.uniform(Config.MIN_DISTANCE, Config.CELL_RADIUS, self.n_users)

        # User positions (x, y)
        self.user_positions = np.column_stack([
            distances * np.cos(angles),
            distances * np.sin(angles)
        ])
        
        # Calculate distances from BS (at origin)
        self.user_distances = np.linalg.norm(self.user_positions, axis=1)
        
        # Calculate channel gains
        self.user_channel_gains = np.array([
            self.channel_model.calculate_channel_gain(dist)
            for dist in self.user_distances
        ])
        
        print(f"Initialized {self.n_users} users")
        print(f"Distance range: {self.user_distances.min():.1f}m - {self.user_distances.max():.1f}m")
        
    def _init_slices(self):
        """Initialize network slices with random user assignment"""
        # Randomly assign users to slices
        # Example: [30, 50, 20] for VoLTE, Video, URLLC
        # ========== For random distribute users =========
        # slice_sizes = self._random_slice_assignment()

        # ========= For Manual Set users ==========
        slice_sizes = Config.SLICE_USER_COUNTS

        user_ids = list(range(self.n_users))
        self.random_state.shuffle(user_ids)
        
        # Split users into slices
        start_idx = 0
        self.slices = {}
        self.user_to_slice = {}  # Maps user_id -> slice_name
        
        for i, slice_name in enumerate(Config.SLICE_NAMES):
            end_idx = start_idx + slice_sizes[i]
            slice_user_ids = user_ids[start_idx:end_idx]
            
            # Create slice
            self.slices[slice_name] = NetworkSlice(
                slice_name,
                slice_user_ids,
                self.channel_model,
                self.random_state,
                env=self
            )
            
            # Record user-to-slice mapping
            for user_id in slice_user_ids:
                self.user_to_slice[user_id] = slice_name
            
            start_idx = end_idx
            print(f"{slice_name} slice: {len(slice_user_ids)} users")

    def _generate_state(self):
        """
        Generates an initial state (assumed to be 0 packets across 3 slices).
        """
        return np.array([0,0,0])

    def _random_slice_assignment(self):
        """
        Randomly assign users to slices
        
        Returns:
            list: Number of users per slice [n_volte, n_video, n_urllc]
        """
        # Generate random proportions
        proportions = self.random_state.dirichlet([1, 1, 1])
        
        # Convert to user counts
        slice_sizes = (proportions * self.n_users).astype(int)
        
        # Adjust to ensure sum equals n_users
        diff = self.n_users - slice_sizes.sum()
        slice_sizes[0] += diff
        
        return slice_sizes.tolist()
    
    def reset(self):
        """
        Reset environment for new episode
        
        Returns:
            np.array: Initial state
        """
        self.current_step = 0
        self.global_slot = 0

        # Reset slice statistics
        for slice_obj in self.slices.values():
            slice_obj.reset_statistics()
        
        # Generate initial state (packet arrivals in first step)
        state = self._generate_state()
        
        return state

    def step(self, action):
        """
        Execute one decision step (400 slots simulation)
        
        Args:
            action (np.array): Bandwidth allocation [w_volte, w_video, w_urllc] in MHz
            
        Returns:
            tuple: (next_state, reward, done, info)
        """

        # Convert action to bandwidth in Hz
        bandwidth_allocation = self._process_action(action)
        
        # Allocate resources to slices
        self._allocate_resources(bandwidth_allocation)

         # Reset slice statistics for this step
        for slice_obj in self.slices.values():
            slice_obj.reset_statistics()

        # Clear all queues at start of each step
        # for slice_obj in self.slices.values():
        #     slice_obj.scheduler.clear_all_queues()
        #     slice_obj.scheduler.current_packet = None

        # Reset step statistics
        self.user_transmitted_bits = {uid: 0 for uid in range(self.n_users)}
        self.user_packets_in_step = {uid: [] for uid in range(self.n_users)}
        
        # Generate traffic for all slices
        all_packets = {}
        start_slot = self.global_slot
        state = []

        for slice_name in Config.SLICE_NAMES:
            slice_obj = self.slices[slice_name]
            packets = slice_obj.generate_traffic(
                duration_slots=Config.SLOTS_PER_STEP,
                start_slot=start_slot
            )
            all_packets[slice_name] = packets

            state.append(len(packets))
            
            # Record packets for rate SLA checking
            for packet in packets:
                self.user_packets_in_step[packet.user_id].append(packet)

        # Simulate 400 slots
        self._simulate_slots(all_packets, start_slot)
        
        # for slice_name in Config.SLICE_NAMES:
        #     slice_obj = self.slices[slice_name]
        #     all_quenue_length = slice_obj.scheduler.get_queue_length()
        #     print(f"{slice_name} queue length: {all_quenue_length}")

        # Check QoS and calculate metrics
        self._check_qos()
        
        # Calculate metrics
        metrics = self._calculate_metrics(bandwidth_allocation)
        
        # Calculate reward
        reward = self._calculate_reward(metrics['utility'])

        # Update step counter
        self.current_step += 1
        self.global_slot += Config.SLOTS_PER_STEP

        term, trunc = False, False  # Continuous task
        
        info = metrics

        # print(f"Step {self.current_step}:")
        # print(f"  SE: {metrics['spectrum_efficiency']:.6f}")
        # print(f"  SSR: {metrics['ssr']}")
        # print(f"  Utility: {metrics['utility']:.6f}")
        # print(f"  Reward: {reward:.6f}")
        # print(f"  Action: {action}")

        return state, reward, term, trunc, info

    def _process_action(self, action):
        """
        Process action to ensure valid bandwidth allocation
        
        Args:
            action (np.array or list): Bandwidth allocation in MHz
            
        Returns:
            dict: Bandwidth allocation per slice in Hz
        """
        action = np.array(action)
        
        # Ensure non-negative
        action = np.maximum(action, 0)
        
        # Normalize to total bandwidth if sum exceeds
        total = action.sum()
        if total > Config.TOTAL_BANDWIDTH / 1e6:  # action in MHz
            action = action * (Config.TOTAL_BANDWIDTH / 1e6) / total
        
        # Convert to Hz
        bandwidth_allocation = {}
        for i, slice_name in enumerate(Config.SLICE_NAMES):
            bandwidth_allocation[slice_name] = action[i] * 1e6  # MHz to Hz
        
        return bandwidth_allocation
    
    def _allocate_resources(self, bandwidth_allocation):
        """
        Allocate bandwidth and power to slices
        
        Args:
            bandwidth_allocation (dict): Bandwidth per slice in Hz
        """
        for slice_name, bandwidth in bandwidth_allocation.items():
            slice_obj = self.slices[slice_name]
            slice_obj.bandwidth = bandwidth
            
            # Allocate power proportional to bandwidth
            bandwidth_ratio = bandwidth / Config.TOTAL_BANDWIDTH
            slice_obj.power = Config.P_TOTAL_W * bandwidth_ratio
    
    def _simulate_slots(self, all_packets, start_slot):
        """
        Simulate 400 time slots
        
        Args:
            all_packets (dict): Pre-generated packets for each slice
        """
        end_slot = start_slot + Config.SLOTS_PER_STEP

        # Create arrival schedule
        arrival_schedule = {slot: [] for slot in range(start_slot, end_slot)}
        
        for slice_name, packets in all_packets.items():
            for packet in packets:
                if start_slot <= packet.arrival_time < end_slot:
                    arrival_schedule[packet.arrival_time].append(packet)

        # Simulate each slot
        for slot in range(start_slot, end_slot):
            # Step 1: Handle packet arrivals
            for packet in arrival_schedule[slot]:
                slice_obj = self.slices[packet.slice_name]
                slice_obj.enqueue_packet(packet)
            
            # Step 2: Periodic timeout check
            if slot % Config.TIMEOUT_CHECK_INTERVAL == 0:
                for slice_name in Config.SLICE_NAMES:
                    slice_obj = self.slices[slice_name]
                    dropped_packets = slice_obj.scheduler.check_and_drop_expired(slot)
                    # Update dropped packet count
                    slice_obj.dropped_packets += len(dropped_packets)

            #  Step 3: Transmit packets from each slice
            for slice_name in Config.SLICE_NAMES:
                self._transmit_slice(slice_name, slot)


    
    def _transmit_slice(self, slice_name, slot):
        """
        Transmit packets for one slice in one slot.

        Uses the full slot bit budget (determined by bandwidth allocation) to
        serve as many queued packets as possible, so that bandwidth allocation
        has a measurable effect on SSR and reward.

        Args:
            slice_name (str): Slice name ('VoLTE', 'Video', or 'URLLC')
            slot (int): Current global slot number
        """
        slice_obj = self.slices[slice_name]
        scheduler = slice_obj.scheduler

        n_users = len(slice_obj.user_ids)
        if n_users == 0:
            return

        user_power = slice_obj.power / n_users

        # Use average channel gain to estimate how many packets this slice can serve per slot
        avg_gain = float(np.mean([self.user_channel_gains[uid] for uid in slice_obj.user_ids]))
        avg_bits_per_slot = self.channel_model.calculate_transmittable_bits(
            avg_gain, user_power, slice_obj.bandwidth, Config.SLOT_DURATION
        )
        min_pkt_size   = Config.get_packet_size(slice_name)
        pkts_remaining = int(avg_bits_per_slot / min_pkt_size) + 1

        # Serve packets until the estimated slot capacity is used up or all queues are empty.
        while pkts_remaining > 0:
            # get_next_packet returns the in-progress packet
            packet, dropped = scheduler.get_next_packet(current_slot=slot)
            slice_obj.dropped_packets += len(dropped)

            if packet is None:
                break   # All queues empty; unused bandwidth is wasted this slot.

            packet.start_transmission(slot)

            # Compute deliverable bits for this specific user using their own channel gain.
            user_bits = self.channel_model.calculate_transmittable_bits(
                self.user_channel_gains[packet.user_id],
                user_power,
                slice_obj.bandwidth,
                Config.SLOT_DURATION
            )

            completed = scheduler.transmit(user_bits)

            if completed is not None:
                # Packet fully delivered: record bits and move to next packet.
                completed.complete_transmission(slot)
                self.user_transmitted_bits[completed.user_id] += completed.original_size
                pkts_remaining -= 1
            else:
                # Packet spans multiple slots; stop here for this slot.
                break



    def _check_qos(self):
        """
        Check QoS for all packets with per-slice aggregate rate checking
        """
        user_rate_satisfied = {}
        
        for user_id in range(self.n_users):
            slice_name = self.user_to_slice[user_id]
            
            bits_transmitted = self.user_transmitted_bits[user_id]
            user_rate = bits_transmitted / Config.STEP_DURATION  # bps
            required_rate = Config.get_rate_sla(slice_name)
            
            user_rate_satisfied[user_id] = (user_rate >= required_rate)
        
        for user_id in range(self.n_users):
            packets = self.user_packets_in_step[user_id]
            if len(packets) == 0:
                continue
            
            slice_name = self.user_to_slice[user_id]
            slice_obj = self.slices[slice_name]
            
            rate_satisfied = user_rate_satisfied[user_id]
            
            for packet in packets:
                x_n_k = 0
                
                if not rate_satisfied:
                    x_n_k = 0
                else:
                    if packet.transmission_complete is not None:
                        latency_satisfied = packet.is_deadline_met()
                        x_n_k = 1 if latency_satisfied else 0
                    else:
                        x_n_k = 0
                
                if x_n_k == 1:
                    slice_obj.total_successful_packets += 1

    def _calculate_metrics(self, bandwidth_allocation):
        """
        Calculate performance metrics
        
        Args:
            bandwidth_allocation (dict): Bandwidth allocation
            
        Returns:
            dict: Performance metrics
        """
        # Calculate Spectrum Efficiency (SE)

        # =========== Use actual throughput to calcualte SE ========
        total_rate = 0
        for slice_name in Config.SLICE_NAMES:
            slice_obj = self.slices[slice_name]
            for user_id in slice_obj.user_ids:
                bits_transmitted = self.user_transmitted_bits[user_id]
                rate = bits_transmitted / Config.STEP_DURATION
                total_rate += rate
        
        spectrum_efficiency = total_rate / Config.TOTAL_BANDWIDTH


        # Calculate SSR for each slice
        ssrs = {}
        for slice_name in Config.SLICE_NAMES:
            ssrs[slice_name] = self.slices[slice_name].get_ssr()

        # Calculate utility
        utility = Config.ALPHA * spectrum_efficiency
        for i, slice_name in enumerate(Config.SLICE_NAMES):
            utility += Config.BETA[i] * ssrs[slice_name]
        
        # Collect statistics
        metrics = {
            'spectrum_efficiency': spectrum_efficiency,
            'ssr': ssrs,
            'utility': utility,
            'bandwidth_allocation': bandwidth_allocation,
            'total_arrived_packets': {
                slice_name: self.slices[slice_name].total_arrived_packets
                for slice_name in Config.SLICE_NAMES
            },
            'total_successful_packets': {
                slice_name: self.slices[slice_name].total_successful_packets
                for slice_name in Config.SLICE_NAMES
            },
            'dropped_packets': {
                slice_name: self.slices[slice_name].dropped_packets
                for slice_name in Config.SLICE_NAMES
            }
        }
        
        return metrics
    
    def _calculate_reward(self, utility):
        """
        Calculate reward with shaping (equation 6 from paper)
        
        Args:
            utility (float): Utility value
            
        Returns:
            float: Shaped reward
        """
        # Calculate growing threshold

        #  =========== Reward Test 1 =============
        gamma_t = min(
            Config.GAMMA_MIN + Config.GAMMA_GROWTH_RATE * self.current_step,
            Config.GAMMA_MAX
        )
        
        if utility < gamma_t:
            reward = 0.0
        elif gamma_t <= utility < Config.GAMMA_MAX:
            reward = 1.0
        else:
            reward = 1.0 + (utility - Config.GAMMA_MAX) / 2.0
        
        return reward
    
