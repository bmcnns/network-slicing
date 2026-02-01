import numpy as np
from config import Config
from channel import ChannelModel
from network_slice import NetworkSlice
from packet import Packet

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
        distances = self.random_state.uniform(0, Config.CELL_RADIUS, self.n_users)
        
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
        slice_sizes = self._random_slice_assignment()

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
                self.random_state
            )
            
            # Record user-to-slice mapping
            for user_id in slice_user_ids:
                self.user_to_slice[user_id] = slice_name
            
            start_idx = end_idx
            print(f"{slice_name} slice: {len(slice_user_ids)} users")

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

        # DEBUG: check action
        # print("Action After Setting:")
        # for slice_name in Config.SLICE_NAMES:
        #     slice_obj = self.slices[slice_name]
        #     print(f"{slice_obj.slice_name} allocation bandwidth: {slice_obj. bandwidth}")            

        # Clear all queues at start of each step
        # for slice_name in Config.SLICE_NAMES:
        #     slice_obj = self.slices[slice_name]
        #     slice_obj.scheduler.queue.clear()
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

            # if len(packets) > 0:
            #     # Show first 3 packets
            #     print(f"    First 3 packets:")
            #     for i, pkt in enumerate(packets[:3]):
            #         print(f"      [{i}] arrival={pkt.arrival_time}, deadline={pkt.deadline}, "
            #             f"user={pkt.user_id}, size={pkt.size}")
                
            #     # Show last 3 packets
            #     if len(packets) > 3:
            #         print(f"    Last 3 packets:")
            #         for i, pkt in enumerate(packets[-3:], len(packets)-3):
            #             print(f"      [{i}] arrival={pkt.arrival_time}, deadline={pkt.deadline}, "
            #                 f"user={pkt.user_id}, size={pkt.size}")
            
            # Record packets for rate SLA checking
            for packet in packets:
                self.user_packets_in_step[packet.user_id].append(packet)
        
        # DEBUG: checck traffic
        # video_users = [uid for uid, s in self.user_to_slice.items() if s == 'Video']
        # for user_id in video_users:
        #     pkt_count = len(self.user_packets_in_step[user_id])
        #     if pkt_count > 0 or user_id == video_users[0]:  # Show first user even if 0
        #         print(f"  User {user_id}: {pkt_count} packets")

        # DEBUG: checck traffic
        # for slice_name, packets in all_packets.items():
        #     if len(packets) == 0:
        #         print(f"{slice_name}: 0 packets")
        #         continue

        #     packets_sorted = sorted(packets, key=lambda p: p.arrival_time)

        #     first_pkt = packets_sorted[0]
        #     last_pkt = packets_sorted[-1]

        #     print(
        #         f"{slice_name}: "
        #         f"total={len(packets)}, "
        #         f"first(arrival={first_pkt.arrival_time}, size={first_pkt.original_size}, user={first_pkt.user_id}), "
        #         f"last(arrival={last_pkt.arrival_time}, size={last_pkt.original_size}, user={last_pkt.user_id})"
        #     )

        # Simulate 400 slots
        self._simulate_slots(all_packets, start_slot)
        
        # Check QoS and calculate metrics
        self._check_qos()
        
        # Calculate metrics
        metrics = self._calculate_metrics(bandwidth_allocation)
        
        # Calculate reward
        reward = self._calculate_reward(metrics['utility'])
                
        # Update step counter
        self.current_step += 1
        self.global_slot += Config.SLOTS_PER_STEP

        done = False  # Continuous task
        
        info = metrics
        
        return state, reward, done, info
    
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
                if packet.arrival_time < Config.SLOTS_PER_STEP:
                    arrival_schedule[packet.arrival_time].append(packet)

        # DEBUG: check slot
        # for i in range(Config.SLOTS_PER_STEP):
        #     if i < 10 and len(arrival_schedule[i]) > 0:
        #         print(f"\n  Slot {i} (DETAILED):")
        #         print(f"    Total: {len(arrival_schedule[i])} packets")
                
        #         slice_counts = {'VoLTE': 0, 'Video': 0, 'URLLC': 0}
        #         for pkt in arrival_schedule[i]:
        #             slice_counts[pkt.slice_name] += 1
                
        #         for slice_name, count in slice_counts.items():
        #             percentage = count / len(arrival_schedule[i]) * 100
        #             print(f"    {slice_name}: {count:3d} packets ({percentage:5.1f}%)")
            # print(f"slot {i}: {len(arrival_schedule[i])}")

        # Simulate each slot
        for slot in range(start_slot, end_slot):
        # for slot in range(25):

            # Handle packet arrivals
            for packet in arrival_schedule[slot]:
                slice_obj = self.slices[packet.slice_name]
                slice_obj.enqueue_packet(packet)
            
            # DEBUG: check queue
            # for slice_name in Config.SLICE_NAMES:
            #     slice_obj = self.slices[slice_name]
            #     print(f"slice {slice_name}: {len(slice_obj.scheduler.queue)}")
            # print()

            # Transmit packets from each slice
            for slice_name in Config.SLICE_NAMES:
                self._transmit_slice(slice_name, slot)

    def _transmit_slice(self, slice_name, slot):
        """
        Transmit packets for one slice in one slot
        
        Args:
            slice_name (str): Slice name
            slot (int): Current slot number
        """
        slice_obj = self.slices[slice_name]
        scheduler = slice_obj.scheduler
        
        # Get next packet to transmit
        packet = scheduler.get_next_packet()
        if packet is None:
            return  # No packets to transmit

        # Mark transmission start
        packet.start_transmission(slot)
        
        # Get user's channel gain
        user_id = packet.user_id
        channel_gain = self.user_channel_gains[user_id]
        
        # Calculate per-user power (equal power per user in slice)
        n_users_in_slice = len(slice_obj.user_ids)
        user_power = slice_obj.power / n_users_in_slice if n_users_in_slice > 0 else 0
        
        # Calculate how many bits can be transmitted this slot
        bits_transmittable = self.channel_model.calculate_transmittable_bits(
            channel_gain,
            user_power,
            slice_obj.bandwidth,
            Config.SLOT_DURATION
        )
        
        # Transmit bits
        completed_packet = scheduler.transmit(bits_transmittable)
        
        # If packet completed
        if completed_packet is not None:
            completed_packet.complete_transmission(slot)
            
            # Record transmitted bits for rate SLA
            self.user_transmitted_bits[user_id] += completed_packet.original_size

    def _check_qos(self):
        """
        Check QoS for all packets with per-slice aggregate rate checking
        """
        # ============================================
        # Step 1: Calculate per-slice aggregate transmitted bits
        # ============================================
        slice_transmitted_bits = {
            'VoLTE': 0,
            'Video': 0,
            'URLLC': 0
        }
        
        for user_id in range(self.n_users):
            slice_name = self.user_to_slice[user_id]
            slice_transmitted_bits[slice_name] += self.user_transmitted_bits[user_id]
                
        # ============================================
        # Step 2: Check per-slice aggregate rate SLA
        # ============================================
        slice_rate_satisfied = {}
        
        for slice_name in Config.SLICE_NAMES:
            total_bits = slice_transmitted_bits[slice_name]
            aggregate_rate = total_bits / Config.STEP_DURATION  # bps
            required_rate = Config.get_rate_sla(slice_name)
            
            slice_rate_satisfied[slice_name] = (aggregate_rate >= required_rate)
            
            # DEBUG output
            # if self.current_step <= 2:
            #     print(f"\n  {slice_name}:")
            #     print(f"    Transmitted bits: {total_bits:>12,.0f} bits")
            #     print(f"    Aggregate rate:   {aggregate_rate/1e6:>10.2f} Mbps")
            #     print(f"    Required rate:    {required_rate/1e6:>10.2f} Mbps")
            #     print(f"    Bandwidth alloc:  {self.slices[slice_name].bandwidth/1e6:>10.2f} MHz")
            #     print(f"    Status: {'✓ PASS' if slice_rate_satisfied[slice_name] else '✗ FAIL'}")
        
        # ============================================
        # Step 3: Check each user's packets
        # ============================================
        # Count for debugging
        debug_counts = {
            'VoLTE': {'total': 0, 'transmitted': 0, 'latency_pass': 0, 'latency_fail': 0, 'not_transmitted': 0},
            'Video': {'total': 0, 'transmitted': 0, 'latency_pass': 0, 'latency_fail': 0, 'not_transmitted': 0},
            'URLLC': {'total': 0, 'transmitted': 0, 'latency_pass': 0, 'latency_fail': 0, 'not_transmitted': 0}
        }
        
        for user_id in range(self.n_users):
            packets = self.user_packets_in_step[user_id]
            if len(packets) == 0:
                continue
            
            slice_name = self.user_to_slice[user_id]
            slice_obj = self.slices[slice_name]
            
            rate_satisfied = slice_rate_satisfied[slice_name]
            
            for packet in packets:
                debug_counts[slice_name]['total'] += 1
                
                x_n_k = 0
                
                if not rate_satisfied:
                    x_n_k = 0
                else:
                    if packet.transmission_complete is not None:
                        debug_counts[slice_name]['transmitted'] += 1
                        latency_satisfied = packet.is_deadline_met()
                        if latency_satisfied:
                            debug_counts[slice_name]['latency_pass'] += 1
                            x_n_k = 1
                        else:
                            debug_counts[slice_name]['latency_fail'] += 1
                            x_n_k = 0
                    else:
                        debug_counts[slice_name]['not_transmitted'] += 1
                        x_n_k = 0
                
                if x_n_k == 1:
                    slice_obj.total_successful_packets += 1
        
        # ============================================
        # DEBUG: Print packet statistics
        # ============================================
        # if self.current_step <= 2:
        #     for slice_name in Config.SLICE_NAMES:
        #         counts = debug_counts[slice_name]
        #         print(f"\n  {slice_name}:")
        #         print(f"    Total packets:       {counts['total']:>6}")
        #         print(f"    Transmitted:         {counts['transmitted']:>6}")
        #         print(f"    Not transmitted:     {counts['not_transmitted']:>6}")
        #         print(f"    Latency pass:        {counts['latency_pass']:>6}")
        #         print(f"    Latency fail:        {counts['latency_fail']:>6}")
        #         if counts['total'] > 0:
        #             success_rate = counts['latency_pass'] / counts['total']
        #             print(f"    Success rate:        {success_rate:>6.1%}")

    def _calculate_metrics(self, bandwidth_allocation):
        """
        Calculate performance metrics
        
        Args:
            bandwidth_allocation (dict): Bandwidth allocation
            
        Returns:
            dict: Performance metrics
        """
        # Calculate Spectrum Efficiency (SE)
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