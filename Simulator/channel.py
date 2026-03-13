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