import numpy as np
from .config import Config

class ChannelModel:
    """
    Physical layer channel model
    """
    
    def __init__(self):
        """Initialize channel model with config parameters"""
        self.pl_intercept = Config.PL_INTERCEPT
        self.pl_slope = Config.PL_SLOPE
        self.min_distance = Config.MIN_DISTANCE
        self.n0_effective = Config.N0_EFFECTIVE_W_HZ
    
    def calculate_path_loss(self, distance):
        """
        Calculate path loss using 3GPP Urban Macro model
        PL(dB) = 128.1 + 37.6 * log10(d[km])
        
        Args:
            distance (float): Distance in meters
            
        Returns:
            float: Path loss in dB
        """
        # Enforce minimum distance
        if distance < self.min_distance:
            distance = self.min_distance
        
        distance_km = distance / 1000.0
        path_loss_db = self.pl_intercept + self.pl_slope * np.log10(distance_km)
        return path_loss_db
    
    def calculate_channel_gain(self, distance):
        """
        Calculate channel gain (linear scale)
        
        Args:
            distance (float): Distance in meters
            
        Returns:
            float: Channel gain (linear, not dB)
        """
        path_loss_db = self.calculate_path_loss(distance)
        channel_gain = 10 ** (-path_loss_db / 10)
        return channel_gain
    
    def calculate_channel_gain_with_shadowing(self, distance, shadowing_db):
        """
        Calculate channel gain including path loss and shadowing (Hua et al. g_un)
        g_un = path_loss_gain * 10^(S/10),  S ~ N(0, SHADOWING_STD_DB²)
 
        Args:
            distance (float): Distance in meters
            shadowing_db (float): Shadowing sample in dB, drawn from N(0, σ²)
 
        Returns:
            float: Channel gain including path loss and shadowing (linear)
        """
        path_loss_db = self.calculate_path_loss(distance)
        total_loss_db = path_loss_db - shadowing_db  # shadowing_db > 0 means less loss
        return 10 ** (-total_loss_db / 10)
    
    def calculate_snr(self, channel_gain, tx_power, bandwidth):
        """
        Calculate Signal-to-Noise Ratio
        SNR = (g * P) / (N0 * BW)
        
        Args:
            channel_gain (float): Channel gain (linear)
            tx_power (float): Transmit power in Watts
            bandwidth (float): Bandwidth in Hz
            
        Returns:
            float: SNR (linear, not dB)
        """
        if bandwidth <= 0 or tx_power <= 0:
            return 0.0
        received_power = channel_gain * tx_power
        noise_power = self.n0_effective * bandwidth
        snr = received_power / noise_power
        return snr
    
    def calculate_shannon_rate(self, channel_gain, tx_power, bandwidth):
        """
        Calculate Shannon capacity
        Rate = BW * log2(1 + SNR)
        
        Args:
            channel_gain (float): Channel gain (linear)
            tx_power (float): Transmit power in Watts
            bandwidth (float): Bandwidth in Hz
            
        Returns:
            float: Data rate in bits per second
        """
        snr = self.calculate_snr(channel_gain, tx_power, bandwidth)
        rate = bandwidth * np.log2(1 + snr)
        return rate
    
    def apply_rayleigh_fading(self, path_loss_gain, random_state):
        """
        Apply Rayleigh fast fading on top of path loss
                
        Args:
            path_loss_gain (float): Static path-loss channel gain
            random_state (np.random.RandomState): RNG for reproducibility
            
        Returns:
            float: Instantaneous channel gain including fading
        """
        h = random_state.exponential(1.0)   # Rayleigh power ~ Exp(1)
        return path_loss_gain * h
 
    def apply_rayleigh_fading_batch(self, path_loss_gains, random_state):
        """
        Vectorised version: apply independent Rayleigh fading to all users.
        
        Args:
            path_loss_gains (np.ndarray): Static path-loss gains, shape (N,)
            random_state (np.random.RandomState): RNG
            
        Returns:
            np.ndarray: Instantaneous gains, shape (N,)
        """
        h = random_state.exponential(1.0, size=len(path_loss_gains))
        return path_loss_gains * h
    
    def calculate_transmittable_bits(self, channel_gain, tx_power, 
                                    bandwidth, duration):
        """
        Calculate how many bits can be transmitted in a given duration
        
        Args:
            channel_gain (float): Channel gain
            tx_power (float): Transmit power in Watts
            bandwidth (float): Bandwidth in Hz
            duration (float): Duration in seconds
            
        Returns:
            float: Number of bits that can be transmitted
        """
        rate = self.calculate_shannon_rate(channel_gain, tx_power, bandwidth)
        bits = rate * duration
        return bits