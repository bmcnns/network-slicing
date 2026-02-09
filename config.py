class Config:
    """System configuration parameters"""
    
    # ============================================
    # Cell and Users
    # ============================================
    N_USERS = 100  # Total number of users
    CELL_RADIUS = 40  # meters
    N_SLICES = 3  # VoLTE, Video, URLLC
    
    # Initial user distribution per slice (will be randomly generated)
    # Example: [30, 50, 20] means 30 VoLTE, 50 Video, 20 URLLC
    SLICE_NAMES = ['VoLTE', 'Video', 'URLLC']
    
    # ============================================
    # Bandwidth and Allocation
    # ============================================
    TOTAL_BANDWIDTH = 20e6  # 20 MHz in Hz
    BANDWIDTH_UNIT = 1e6  # 1 MHz allocation granularity
    MIN_BANDWIDTH_PER_SLICE = 1e6  # At least 1 MHz per slice
    
    # ============================================
    # Time Structure
    # ============================================
    SLOT_DURATION = 0.5e-3  # 0.5 ms in seconds
    SLOTS_PER_STEP = 400  # One decision step = 400 slots = 200 ms
    STEP_DURATION = SLOT_DURATION * SLOTS_PER_STEP  # 0.2 seconds
    
    # ============================================
    # Physical Layer - 3GPP TR 36.814 Parameters
    # ============================================
    
    # Path loss model: Urban Macro
    # PL(dB) = 128.1 + 37.6 * log10(d[km])
    PL_INTERCEPT = 128.1  # dB
    PL_SLOPE = 37.6  # dB per decade
    MIN_DISTANCE = 35  # meters (model validity)
    CARRIER_FREQ = 2.0e9  # 2 GHz
    
    # Power parameters
    P_TOTAL_DBM = 46  # dBm = 40W (macro BS)
    P_TOTAL_W = 10 ** ((P_TOTAL_DBM - 30) / 10)  # Convert to Watts
    
    # Noise parameters
    N0_DBM_HZ = -174  # dBm/Hz (thermal noise)
    NOISE_FIGURE_DB = 9  # dB (receiver noise figure)
    N0_EFFECTIVE_DBM_HZ = N0_DBM_HZ + NOISE_FIGURE_DB  # -165 dBm/Hz
    N0_EFFECTIVE_W_HZ = 10 ** ((N0_EFFECTIVE_DBM_HZ - 30) / 10)
    
    # Power allocation strategy: proportional to bandwidth
    POWER_ALLOCATION = 'proportional_to_bandwidth'
    
    # ============================================
    # Traffic Parameters (Based on 3GPP TR 36.814)
    # ============================================
    
    # VoLTE - Uniform distribution
    VOLTE_MIN_INTERVAL = 40  # slots (20 ms)
    VOLTE_MAX_INTERVAL = 80  # slots (40 ms)
    VOLTE_PACKET_SIZE = 1020  # bits (51 Kbps * 20ms)
    
    # Video - Truncated Exponential distribution
    VIDEO_LAMBDA = 0.02  # arrival rate parameter
    VIDEO_MAX_INTERVAL = 100  # slots (50 ms) - truncation limit
    VIDEO_PACKET_SIZE = 1280  # bits (64 Kbps * 20ms)
    
    # URLLC - Exponential distribution
    URLLC_LAMBDA = 0.05  # arrival rate parameter
    URLLC_PACKET_SIZE = 2000  # bits (9 Mbps * 0.5ms, large packet)
    
    # ============================================
    # SLA Requirements (from paper Table I)
    # ============================================
    
    # Rate requirements (bits per second) for per slice
    # RATE_SLA = {
    #    'VoLTE': 51e3,
    #    'Video': 100e6,
    #    'URLLC': 10e6
    # }

    # Rate requirements (bits per second) for per user
    RATE_SLA = {
        'VoLTE': 0.005e6,
        'Video': 0.01e6,
        'URLLC': 0.05e6
    }
    
    # Latency requirements (in slots)
    # LATENCY_SLA = {
    #     'VoLTE': 20,   # 10 ms = 20 slots
    #     'Video': 20,   # 10 ms = 20 slots
    #     'URLLC': 2     # 1 ms = 2 slots
    # }


    LATENCY_SLA = {
        'VoLTE': 50,
        'Video': 50,
        'URLLC': 20
    }
    
    # ============================================
    # Queue Parameters
    # ============================================
    # MAX_QUEUE_SIZE = 1000  # Maximum packets per slice queue
    MAX_QUEUE_SIZE = 50 # Maximum packets per slice user
    QUEUE_DISCIPLINE = 'FIFO'  # First In First Out
    
    # ============================================
    # Scheduler
    # ============================================
    SCHEDULER_TYPE = 'round_robin'
    # When packet not finished in one slot: continue in next slot (Method A)
    PACKET_CONTINUATION = True  
    
    # ============================================
    # Utility Function Parameters
    # ============================================
    ALPHA = 0.1  # Weight for spectrum efficiency
    BETA = [1.0, 1.0, 5.0]  # Weights for SSR of each slice
    
    # Reward shaping parameters (from paper equation 6)
    # GAMMA_MIN = 4.5
    # GAMMA_MAX = 6.5
    # GAMMA_GROWTH_RATE = 0.0001  # Per step

    GAMMA_MIN = 0.6
    GAMMA_MAX = 2.0
    GAMMA_GROWTH_RATE = 0.0001  # Per step

    
    # ============================================
    # Simulation Parameters
    # ============================================
    RANDOM_SEED = 2022  # For reproducibility
    
    @staticmethod
    def get_packet_size(slice_name):
        """Get packet size for a given slice"""
        sizes = {
            'VoLTE': Config.VOLTE_PACKET_SIZE,
            'Video': Config.VIDEO_PACKET_SIZE,
            'URLLC': Config.URLLC_PACKET_SIZE
        }
        return sizes[slice_name]
    
    @staticmethod
    def get_latency_sla(slice_name):
        """Get latency SLA for a given slice (in slots)"""
        return Config.LATENCY_SLA[slice_name]
    
    @staticmethod
    def get_rate_sla(slice_name):
        """Get rate SLA for a given slice (in bps)"""
        return Config.RATE_SLA[slice_name]