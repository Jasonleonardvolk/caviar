import pytest
import asyncio
import websockets
import json
import numpy as np
import time
from typing import Dict, Any, List
import struct

# Test configuration
TEST_HOST = "localhost"
TEST_PORT = 8000
TEST_BASE_URL = f"ws://{TEST_HOST}:{TEST_PORT}/api/v1/ws/audio/ingest"

class WebSocketTestClient:
    """Helper class for WebSocket testing"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.uri = f"{TEST_BASE_URL}?session_id={session_id}"
        self.websocket = None
        self.messages = []
        
    async def __aenter__(self):
        self.websocket = await websockets.connect(self.uri)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.websocket:
            await self.websocket.close()
            
    async def send_audio(self, audio_data: np.ndarray):
        """Send audio data as bytes"""
        if audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        await self.websocket.send(audio_data.tobytes())
        
    async def send_json(self, data: Dict[str, Any]):
        """Send JSON control message"""
        await self.websocket.send(json.dumps(data))
        
    async def receive_json(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Receive and parse JSON message"""
        msg = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
        data = json.loads(msg)
        self.messages.append(data)
        return data
        
    async def wait_for_message(self, msg_type: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Wait for specific message type"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                msg = await self.receive_json(timeout=0.5)
                if msg.get("type") == msg_type:
                    return msg
            except asyncio.TimeoutError:
                continue
        raise TimeoutError(f"Timeout waiting for message type: {msg_type}")

def generate_test_audio(frequency: float = 440, duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test sine wave audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    return (audio * 32767).astype(np.int16)

def generate_noise(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate random noise"""
    samples = int(sample_rate * duration)
    noise = np.random.normal(0, 0.1, samples)
    return (noise * 32767).astype(np.int16)

def generate_chirp(duration: float = 1.0, f0: float = 100, f1: float = 1000, sample_rate: int = 16000) -> np.ndarray:
    """Generate chirp signal (frequency sweep)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    audio = np.sin(phase) * 0.5
    return (audio * 32767).astype(np.int16)

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test basic WebSocket connection"""
    async with WebSocketTestClient("test_connection") as client:
        # Should receive connected message
        msg = await client.receive_json()
        assert msg["type"] == "connected"
        assert msg["session_id"] == "test_connection"
        assert "timestamp" in msg
        assert "config" in msg
        
        # Check config
        config = msg["config"]
        assert config["sample_rate"] == 16000
        assert config["chunk_duration"] == 1.0
        assert config["format"] == "int16"

@pytest.mark.asyncio
async def test_audio_websocket_silent():
    """Test WebSocket with silent audio"""
    async with WebSocketTestClient("test_silent") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Create silent audio
        silent_audio = np.zeros(16000, dtype=np.int16)
        await client.send_audio(silent_audio)
        
        # Should receive silence indicator
        msg = await client.wait_for_message("silence", timeout=2.0)
        assert msg["type"] == "silence"
        
        # Send finalize
        await client.send_json({"type": "finalize"})
        
        # Should get final result
        msg = await client.wait_for_message("final")
        assert msg["type"] == "final"
        assert "data" in msg

@pytest.mark.asyncio
async def test_audio_websocket_speech():
    """Test WebSocket with speech-like audio"""
    async with WebSocketTestClient("test_speech") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Create speech-like audio (mix of frequencies)
        audio = generate_test_audio(200) + generate_test_audio(300) * 0.5 + generate_test_audio(400) * 0.3
        await client.send_audio(audio)
        
        # Should receive partial result
        msg = await client.wait_for_message("partial")
        assert msg["type"] == "partial"
        assert "data" in msg
        
        data = msg["data"]
        assert "spectral" in data
        assert "emotion" in data
        assert "hologram_hint" in data
        
        # Check spectral data
        spectral = data["spectral"]
        assert spectral["centroid"] > 0
        assert 0 <= spectral["rms"] <= 1
        
        # Check emotion data
        emotion = data["emotion"]
        assert emotion["label"] in ["neutral", "calm", "excited", "energetic", "focused"]
        assert 0 <= emotion["confidence"] <= 1
        
        # Check hologram hint
        hologram = data["hologram_hint"]
        assert 0 <= hologram["hue"] <= 360
        assert 0 <= hologram["intensity"] <= 1
        assert 0 <= hologram["psi"] <= 1

@pytest.mark.asyncio
async def test_audio_websocket_malformed():
    """Test WebSocket with malformed data"""
    async with WebSocketTestClient("test_malformed") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Send malformed data (not proper audio format)
        await client.websocket.send(b"not valid audio data")
        
        # Should handle gracefully (might get silence or error)
        msg = await client.receive_json(timeout=2.0)
        assert msg["type"] in ["error", "silence"]

@pytest.mark.asyncio
async def test_audio_websocket_large_chunk():
    """Test WebSocket with large audio chunks"""
    async with WebSocketTestClient("test_large") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Create large audio chunk (5 seconds)
        large_audio = generate_test_audio(duration=5.0)
        await client.send_audio(large_audio)
        
        # Should receive multiple partials
        partials_received = 0
        start_time = time.time()
        
        while time.time() - start_time < 10.0:
            try:
                msg = await client.receive_json(timeout=1.0)
                if msg["type"] == "partial":
                    partials_received += 1
            except asyncio.TimeoutError:
                break
        
        assert partials_received >= 1  # Should process large chunks

@pytest.mark.asyncio
async def test_audio_websocket_concurrent():
    """Test concurrent WebSocket connections"""
    async def client_session(session_id: str, frequency: float):
        async with WebSocketTestClient(session_id) as client:
            # Connect
            await client.wait_for_message("connected")
            
            # Send unique audio
            audio = generate_test_audio(frequency=frequency)
            await client.send_audio(audio)
            
            # Get partial
            msg = await client.wait_for_message("partial")
            assert msg["type"] == "partial"
            
            # Check spectral centroid is roughly correct
            centroid = msg["data"]["spectral"]["centroid"]
            assert abs(centroid - frequency) < 100  # Within 100Hz
            
            # Finalize
            await client.send_json({"type": "finalize"})
            msg = await client.wait_for_message("final")
            assert msg["type"] == "final"
    
    # Run multiple concurrent sessions with different frequencies
    await asyncio.gather(
        client_session("test_concurrent_1", 440),
        client_session("test_concurrent_2", 880),
        client_session("test_concurrent_3", 1320)
    )

@pytest.mark.asyncio
async def test_websocket_ping_pong():
    """Test WebSocket ping/pong mechanism"""
    async with WebSocketTestClient("test_ping") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Send ping
        await client.send_json({"type": "ping"})
        
        # Should receive pong
        msg = await client.wait_for_message("pong")
        assert msg["type"] == "pong"
        assert "timestamp" in msg
        assert "stats" in msg

@pytest.mark.asyncio
async def test_websocket_stats():
    """Test WebSocket statistics"""
    async with WebSocketTestClient("test_stats") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Send some audio
        for i in range(3):
            audio = generate_test_audio(duration=0.5)
            await client.send_audio(audio)
            await asyncio.sleep(0.1)
        
        # Request stats
        await client.send_json({"type": "stats"})
        
        # Should receive stats
        msg = await client.wait_for_message("stats", timeout=5.0)
        assert msg["type"] == "stats"
        assert "data" in msg
        
        stats = msg["data"]
        assert stats["bytes_received"] > 0
        assert stats["chunks_processed"] >= 0

@pytest.mark.asyncio
async def test_websocket_reconnection():
    """Test WebSocket reconnection handling"""
    session_id = "test_reconnect"
    
    # First connection
    async with WebSocketTestClient(session_id) as client1:
        await client1.wait_for_message("connected")
        audio = generate_test_audio()
        await client1.send_audio(audio)
        await client1.wait_for_message("partial")
    
    # Brief pause
    await asyncio.sleep(0.5)
    
    # Second connection with same session ID
    async with WebSocketTestClient(session_id) as client2:
        await client2.wait_for_message("connected")
        audio = generate_test_audio(frequency=880)
        await client2.send_audio(audio)
        msg = await client2.wait_for_message("partial")
        
        # Should work normally
        assert msg["type"] == "partial"

@pytest.mark.asyncio
async def test_audio_websocket_streaming():
    """Test continuous audio streaming"""
    async with WebSocketTestClient("test_streaming") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Stream audio chunks continuously
        chunk_duration = 0.1  # 100ms chunks
        chunks_sent = 0
        partials_received = 0
        
        # Send audio for 2 seconds
        start_time = time.time()
        while time.time() - start_time < 2.0:
            # Generate and send chunk
            audio = generate_test_audio(duration=chunk_duration)
            await client.send_audio(audio)
            chunks_sent += 1
            
            # Check for partials (non-blocking)
            try:
                msg = await client.receive_json(timeout=0.05)
                if msg["type"] == "partial":
                    partials_received += 1
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(chunk_duration)
        
        # Should have received some partials
        assert chunks_sent > 10
        assert partials_received > 0

@pytest.mark.asyncio
async def test_audio_websocket_emotion_changes():
    """Test emotion detection with varying audio"""
    async with WebSocketTestClient("test_emotions") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Send calm audio (low frequency, steady)
        calm_audio = generate_test_audio(frequency=200, duration=1.0)
        await client.send_audio(calm_audio)
        
        msg = await client.wait_for_message("partial")
        emotion1 = msg["data"]["emotion"]["label"]
        
        # Send excited audio (high frequency, noisy)
        excited_audio = generate_chirp(duration=1.0) + generate_noise(duration=1.0) * 0.3
        await client.send_audio(excited_audio)
        
        msg = await client.wait_for_message("partial")
        emotion2 = msg["data"]["emotion"]["label"]
        
        # Emotions might be different (though not guaranteed)
        print(f"Emotions detected: {emotion1} -> {emotion2}")

@pytest.mark.asyncio
async def test_websocket_error_recovery():
    """Test error recovery in WebSocket"""
    async with WebSocketTestClient("test_error_recovery") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Send invalid control message
        await client.send_json({"type": "invalid_type"})
        
        # Should still be able to send audio after error
        audio = generate_test_audio()
        await client.send_audio(audio)
        
        # Should receive partial normally
        msg = await client.wait_for_message("partial")
        assert msg["type"] == "partial"

@pytest.mark.asyncio
async def test_websocket_queue_overflow():
    """Test behavior when processing queue overflows"""
    async with WebSocketTestClient("test_overflow") as client:
        # Wait for connection
        msg = await client.wait_for_message("connected")
        max_queue = msg["config"].get("max_queue_size", 100)
        
        # Send many chunks rapidly
        warnings_received = 0
        for i in range(max_queue + 10):
            audio = generate_noise(duration=0.1)
            await client.send_audio(audio)
            
            # Check for warnings (non-blocking)
            try:
                msg = await client.receive_json(timeout=0.01)
                if msg["type"] == "warning":
                    warnings_received += 1
            except asyncio.TimeoutError:
                pass
        
        # Might receive warnings about dropped chunks
        print(f"Warnings received: {warnings_received}")

# Performance benchmarks
@pytest.mark.asyncio
async def test_websocket_performance():
    """Benchmark WebSocket performance"""
    async with WebSocketTestClient("test_performance") as client:
        # Wait for connection
        await client.wait_for_message("connected")
        
        # Measure throughput
        total_bytes = 0
        start_time = time.time()
        duration = 5.0
        
        while time.time() - start_time < duration:
            audio = generate_test_audio(duration=0.1)
            await client.send_audio(audio)
            total_bytes += audio.nbytes
            
            # Drain messages
            try:
                await client.receive_json(timeout=0.01)
            except asyncio.TimeoutError:
                pass
        
        elapsed = time.time() - start_time
        throughput = total_bytes / elapsed / 1024 / 1024  # MB/s
        
        print(f"WebSocket throughput: {throughput:.2f} MB/s")
        assert throughput > 0.1  # At least 0.1 MB/s

if __name__ == "__main__":
    # Run specific test
    asyncio.run(test_websocket_connection())