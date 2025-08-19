import pathlib
import pytest
import json
import numpy as np
import wave
import struct
from httpx import AsyncClient
from tori_backend.main import app

# Test fixtures directory
FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

def create_test_wav(filename: str, duration: float = 1.0, frequency: int = 440) -> pathlib.Path:
    """Create a valid test WAV file with a sine wave"""
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    
    # Generate sine wave - use endpoint=False to avoid extra sample
    t = np.linspace(0, duration, num_samples, endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write WAV file
    filepath = FIXTURES_DIR / filename
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return filepath

def create_large_test_wav(filename: str, size_mb: int = 101) -> pathlib.Path:
    """Create a valid large WAV file for size testing"""
    sample_rate = 48000
    channels = 2
    sample_width = 2  # 16-bit
    
    # Calculate samples needed for target size
    bytes_per_sample = channels * sample_width
    target_bytes = size_mb * 1024 * 1024
    num_samples = (target_bytes - 44) // bytes_per_sample  # Subtract header size
    
    filepath = FIXTURES_DIR / filename
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        
        # Write in chunks to avoid memory issues
        chunk_size = sample_rate  # 1 second chunks
        silence = np.zeros(chunk_size * channels, dtype=np.int16)
        
        for _ in range(num_samples // chunk_size):
            wav_file.writeframes(silence.tobytes())
        
        # Write remaining samples
        remaining = num_samples % chunk_size
        if remaining > 0:
            wav_file.writeframes(np.zeros(remaining * channels, dtype=np.int16).tobytes())
    
    return filepath

@pytest.fixture
def hello_wav():
    """Create a test audio file"""
    return create_test_wav("hello_test.wav", duration=2.0, frequency=440)

@pytest.fixture
def large_wav():
    """Create a large test audio file"""
    return create_large_test_wav("large_test.wav", size_mb=101)

@pytest.mark.asyncio
async def test_audio_ingest_success(hello_wav):
    """Test successful audio ingestion with full contract"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        with open(hello_wav, 'rb') as f:
            files = {"file": ("hello.wav", f, "audio/wav")}
            response = await ac.post(
                "/api/v1/audio/ingest",
                files=files,
                headers={"client-id": "test_client"}
            )
    
    assert response.status_code == 200
    payload = response.json()
    
    # Core contract assertions
    assert "transcript" in payload
    assert isinstance(payload["transcript"], str)
    
    # Spectral features - using canonical 'spectral' field
    assert "spectral" in payload
    spectral = payload["spectral"]
    assert "spectral_centroid" in spectral
    assert 0.0 <= spectral["spectral_centroid"] < 22050
    assert "rms" in spectral
    assert 0.0 <= spectral["rms"] <= 1.0
    
    # Emotion
    assert "emotion" in payload
    emotion = payload["emotion"]
    assert "excitement" in emotion
    assert "calmness" in emotion
    assert 0.0 <= emotion["excitement"] <= 1.0
    assert 0.0 <= emotion["calmness"] <= 1.0
    
    # Hologram hint
    assert "hologram_hint" in payload
    hh = payload["hologram_hint"]
    assert 0.0 <= hh["hue"] <= 360.0
    assert 0.0 <= hh["intensity"] <= 1.0
    assert 0.0 <= hh["psi"] <= 1.0
    
    # PSI state
    assert "psi_state" in payload
    assert "psi_phase" in payload["psi_state"]
    assert "phase_coherence" in payload["psi_state"]

@pytest.mark.asyncio
async def test_audio_ingest_large_file(large_wav):
    """Test file size limit with valid large WAV"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        with open(large_wav, 'rb') as f:
            files = {"file": ("large.wav", f, "audio/wav")}
            response = await ac.post(
                "/api/v1/audio/ingest",
                files=files,
                headers={"client-id": "test_client"}
            )
    
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()

# Clean up fixtures after tests
@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Clean up test files after all tests
    for file in FIXTURES_DIR.glob("*.wav"):
        file.unlink()