#!/usr/bin/env python3
"""
Python Bridge for Interferometer Hardware
Provides Python interface to Rust interferometer driver via PyO3
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import the compiled Rust module
try:
    import interferometer_driver_rs as rust_driver
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("Rust interferometer driver not available, using mock implementation")


@dataclass
class InterferometerConfig:
    """Interferometer configuration"""
    num_channels: int = 64
    sampling_rate: float = 1e6  # Hz
    integration_time: float = 1e-3  # seconds
    reference_power: float = 1.0  # mW
    quantum_efficiency: float = 0.85


@dataclass
class ChannelReadout:
    """Single channel readout data"""
    channel: int
    amplitude: float
    phase: float
    snr: float
    timestamp: int  # microseconds


class InterferometerBridge:
    """
    Python bridge to interferometer hardware
    
    This class provides a Python-friendly interface to the Rust
    interferometer driver, with fallback to pure Python mock
    implementation for development.
    """
    
    def __init__(self, config: Optional[InterferometerConfig] = None):
        """
        Initialize the interferometer bridge
        
        Args:
            config: Interferometer configuration
        """
        self.config = config or InterferometerConfig()
        
        if RUST_AVAILABLE:
            # Use Rust driver
            self._driver = rust_driver.InterferometerDriver(
                num_channels=self.config.num_channels,
                sampling_rate=self.config.sampling_rate,
                integration_time=self.config.integration_time,
                reference_power=self.config.reference_power,
                quantum_efficiency=self.config.quantum_efficiency
            )
            self._mock = False
        else:
            # Use Python mock
            self._driver = MockInterferometerDriver(self.config)
            self._mock = True
        
        self.is_initialized = False
        self.is_acquiring = False
    
    def initialize(self) -> None:
        """Initialize the hardware"""
        if self.is_initialized:
            raise RuntimeError("Already initialized")
        
        if RUST_AVAILABLE:
            self._driver.initialize()
        else:
            self._driver.initialize()
        
        self.is_initialized = True
        logger.info("Interferometer initialized")
    
    def start_acquisition(self) -> None:
        """Start continuous data acquisition"""
        if not self.is_initialized:
            raise RuntimeError("Not initialized")
        
        if RUST_AVAILABLE:
            self._driver.start_acquisition()
        else:
            self._driver.start_acquisition()
        
        self.is_acquiring = True
        logger.info("Started acquisition")
    
    def stop_acquisition(self) -> None:
        """Stop data acquisition"""
        if not self.is_acquiring:
            return
        
        if RUST_AVAILABLE:
            self._driver.stop_acquisition()
        else:
            self._driver.stop_acquisition()
        
        self.is_acquiring = False
        logger.info("Stopped acquisition")
    
    def read_phase_profile(self) -> np.ndarray:
        """
        Read phase profile from all channels
        
        Returns:
            Array of phase values in radians
        """
        if not self.is_initialized:
            raise RuntimeError("Not initialized")
        
        if RUST_AVAILABLE:
            phases = self._driver.read_phase_profile()
        else:
            phases = self._driver.read_phase_profile()
        
        return np.array(phases)
    
    def read_amplitude_profile(self) -> np.ndarray:
        """
        Read amplitude profile from all channels
        
        Returns:
            Array of amplitude values
        """
        if not self.is_initialized:
            raise RuntimeError("Not initialized")
        
        if RUST_AVAILABLE:
            amplitudes = self._driver.read_amplitude_profile()
        else:
            amplitudes = self._driver.read_amplitude_profile()
        
        return np.array(amplitudes)
    
    def read_complex_field(self) -> np.ndarray:
        """
        Read complex field (amplitude * exp(i*phase))
        
        Returns:
            Complex array representing the field
        """
        amplitudes = self.read_amplitude_profile()
        phases = self.read_phase_profile()
        
        return amplitudes * np.exp(1j * phases)
    
    def read_channel(self, channel: int) -> ChannelReadout:
        """
        Read single channel data
        
        Args:
            channel: Channel index
            
        Returns:
            Channel readout data
        """
        if not self.is_initialized:
            raise RuntimeError("Not initialized")
        
        if channel >= self.config.num_channels:
            raise ValueError(f"Channel {channel} out of range")
        
        if RUST_AVAILABLE:
            data = self._driver.read_channel(channel)
            return ChannelReadout(
                channel=data.channel,
                amplitude=data.amplitude,
                phase=data.phase,
                snr=data.snr,
                timestamp=data.timestamp
            )
        else:
            return self._driver.read_channel(channel)
    
    def set_reference_power(self, power_mw: float) -> None:
        """
        Set reference beam power
        
        Args:
            power_mw: Power in milliwatts (0-10)
        """
        if not 0 <= power_mw <= 10:
            raise ValueError("Power must be between 0 and 10 mW")
        
        if RUST_AVAILABLE:
            self._driver.set_reference_power(power_mw)
        else:
            self._driver.set_reference_power(power_mw)
        
        self.config.reference_power = power_mw
    
    def calibrate_channel(self, channel: int) -> None:
        """Calibrate a specific channel"""
        if not self.is_initialized:
            raise RuntimeError("Not initialized")
        
        if RUST_AVAILABLE:
            self._driver.calibrate_channel(channel)
        else:
            self._driver.calibrate_channel(channel)
        
        logger.info(f"Calibrated channel {channel}")
    
    def calibrate_all(self) -> None:
        """Calibrate all channels"""
        for i in range(self.config.num_channels):
            self.calibrate_channel(i)
    
    def get_status(self) -> Dict[str, str]:
        """Get hardware status"""
        if RUST_AVAILABLE:
            return dict(self._driver.get_status())
        else:
            return self._driver.get_status()
    
    def shutdown(self) -> None:
        """Shutdown the hardware"""
        if self.is_acquiring:
            self.stop_acquisition()
        
        if RUST_AVAILABLE:
            self._driver.shutdown()
        else:
            self._driver.shutdown()
        
        self.is_initialized = False
        logger.info("Interferometer shutdown")
    
    # High-level analysis functions
    
    def measure_soliton_position(self) -> Tuple[float, float]:
        """
        Measure soliton position and width
        
        Returns:
            (position, width) in channel units
        """
        amplitude = self.read_amplitude_profile()
        
        # Find peak
        peak_idx = np.argmax(amplitude)
        peak_amp = amplitude[peak_idx]
        
        # Estimate width at half maximum
        half_max = peak_amp / 2
        above_half = amplitude > half_max
        
        if np.any(above_half):
            indices = np.where(above_half)[0]
            width = indices[-1] - indices[0]
        else:
            width = 1.0
        
        # Refine position with quadratic fit around peak
        if 0 < peak_idx < len(amplitude) - 1:
            y0, y1, y2 = amplitude[peak_idx-1:peak_idx+2]
            offset = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
            position = peak_idx + offset
        else:
            position = float(peak_idx)
        
        return position, float(width)
    
    def measure_phase_gradient(self) -> float:
        """
        Measure average phase gradient (related to soliton velocity)
        
        Returns:
            Phase gradient in radians/channel
        """
        phases = self.read_phase_profile()
        
        # Unwrap phases
        unwrapped = np.unwrap(phases)
        
        # Linear fit
        x = np.arange(len(phases))
        gradient = np.polyfit(x, unwrapped, 1)[0]
        
        return gradient
    
    def detect_dark_solitons(self, threshold: float = 0.5) -> List[int]:
        """
        Detect dark soliton positions
        
        Args:
            threshold: Detection threshold
            
        Returns:
            List of channel indices with dark solitons
        """
        amplitude = self.read_amplitude_profile()
        
        # Dark solitons are amplitude dips
        mean_amp = np.mean(amplitude)
        
        # Find local minima below threshold
        dark_positions = []
        
        for i in range(1, len(amplitude) - 1):
            if (amplitude[i] < amplitude[i-1] and 
                amplitude[i] < amplitude[i+1] and
                amplitude[i] < threshold * mean_amp):
                dark_positions.append(i)
        
        return dark_positions


class MockInterferometerDriver:
    """
    Pure Python mock implementation for development
    """
    
    def __init__(self, config: InterferometerConfig):
        self.config = config
        self.is_initialized = False
        self.is_acquiring = False
        
        # Mock state
        self.channel_amplitudes = np.random.uniform(0.1, 1.0, config.num_channels)
        self.channel_phases = np.random.uniform(0, 2*np.pi, config.num_channels)
        
    def initialize(self):
        import time
        time.sleep(0.5)  # Simulate hardware init
        self.is_initialized = True
    
    def start_acquisition(self):
        self.is_acquiring = True
    
    def stop_acquisition(self):
        self.is_acquiring = False
    
    def read_phase_profile(self) -> List[float]:
        # Add some noise
        noise = np.random.normal(0, 0.01, self.config.num_channels)
        return (self.channel_phases + noise).tolist()
    
    def read_amplitude_profile(self) -> List[float]:
        # Add some noise
        noise = np.random.normal(0, 0.005, self.config.num_channels)
        return np.clip(self.channel_amplitudes + noise, 0, None).tolist()
    
    def read_channel(self, channel: int) -> ChannelReadout:
        import time
        
        amplitude = self.channel_amplitudes[channel]
        phase = self.channel_phases[channel]
        snr = 20 * np.log10(amplitude / 0.01)
        
        return ChannelReadout(
            channel=channel,
            amplitude=amplitude + np.random.normal(0, 0.005),
            phase=phase + np.random.normal(0, 0.01),
            snr=snr,
            timestamp=int(time.time() * 1e6)
        )
    
    def set_reference_power(self, power_mw: float):
        self.config.reference_power = power_mw
    
    def calibrate_channel(self, channel: int):
        import time
        time.sleep(0.1)  # Simulate calibration
    
    def get_status(self) -> Dict[str, str]:
        return {
            'initialized': str(self.is_initialized),
            'acquiring': str(self.is_acquiring),
            'mock': 'true',
            'channels': str(self.config.num_channels)
        }
    
    def shutdown(self):
        self.is_initialized = False
        self.is_acquiring = False
    
    def inject_soliton(self, position: int, width: float, amplitude: float):
        """Inject a mock soliton for testing"""
        x = np.arange(self.config.num_channels)
        envelope = amplitude * np.exp(-(x - position)**2 / (2 * width**2))
        phase = np.pi * (x - position) / width
        
        self.channel_amplitudes = envelope
        self.channel_phases = phase


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create interferometer
    config = InterferometerConfig(num_channels=128)
    interferometer = InterferometerBridge(config)
    
    # Initialize
    interferometer.initialize()
    
    # Inject test soliton (if mock)
    if interferometer._mock:
        interferometer._driver.inject_soliton(64, 10, 0.8)
    
    # Read data
    amplitudes = interferometer.read_amplitude_profile()
    phases = interferometer.read_phase_profile()
    
    print(f"Max amplitude: {np.max(amplitudes):.3f}")
    print(f"Phase range: {np.ptp(phases):.3f} rad")
    
    # Measure soliton
    position, width = interferometer.measure_soliton_position()
    print(f"Soliton position: {position:.1f}, width: {width:.1f}")
    
    # Shutdown
    interferometer.shutdown()
