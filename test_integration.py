#!/usr/bin/env python3
"""Integration test for WebSocket streaming with comprehensive checks"""

import asyncio
import sys
import subprocess
import time
import requests
import websockets
import json
import numpy as np
from typing import Dict, Any, List, Optional
import argparse
from datetime import datetime
import os

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

async def check_server_health() -> bool:
    """Check if server is running and healthy"""
    print_header("Checking Server Health")
    
    endpoints = [
        ("http://localhost:8000/", "Root endpoint"),
        ("http://localhost:8000/health", "Health check"),
        ("http://localhost:8000/info", "API info"),
        ("http://localhost:8000/api/v1/ws/audio/status", "WebSocket status")
    ]
    
    all_healthy = True
    
    for url, description in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print_success(f"{description}: {response.status_code}")
                
                # Print relevant info from responses
                if url.endswith("/health"):
                    data = response.json()
                    status = data.get("status", "unknown")
                    if status != "healthy":
                        print_warning(f"Health status: {status}")
                        if data.get("issues"):
                            for issue in data["issues"]:
                                print_warning(f"  - {issue}")
                                
                elif url.endswith("/info"):
                    data = response.json()
                    print_info(f"API Version: {data['api']['version']}")
                    print_info(f"WebSocket enabled: {data['capabilities']['websocket']['enabled']}")
                    
            else:
                print_error(f"{description}: {response.status_code}")
                all_healthy = False
                
        except requests.exceptions.RequestException as e:
            print_error(f"{description}: {str(e)}")
            all_healthy = False
    
    return all_healthy

def generate_test_audio(frequency: float = 440, duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test sine wave audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    return (audio * 32767).astype(np.int16)

def generate_complex_audio(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate complex audio with multiple frequencies"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Mix of frequencies simulating speech
    audio = (
        np.sin(2 * np.pi * 200 * t) * 0.4 +  # Fundamental
        np.sin(2 * np.pi * 400 * t) * 0.3 +  # Harmonic
        np.sin(2 * np.pi * 600 * t) * 0.2 +  # Harmonic
        np.random.normal(0, 0.05, len(t))    # Noise
    )
    return (audio * 16000).astype(np.int16)

async def test_websocket_connection() -> bool:
    """Test basic WebSocket connection"""
    print_header("Testing WebSocket Connection")
    
    uri = "ws://localhost:8000/api/v1/ws/audio/ingest?session_id=integration_basic"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for connection message
            msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(msg)
            
            if data["type"] == "connected":
                print_success("WebSocket connected")
                print_info(f"Session ID: {data['session_id']}")
                
                # Check configuration
                config = data.get("config", {})
                print_info(f"Sample rate: {config.get('sample_rate', 'N/A')} Hz")
                print_info(f"Chunk duration: {config.get('chunk_duration', 'N/A')} s")
                print_info(f"Format: {config.get('format', 'N/A')}")
                
                return True
            else:
                print_error(f"Unexpected message type: {data['type']}")
                return False
                
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")
        return False

async def test_audio_streaming() -> bool:
    """Test audio streaming with analysis"""
    print_header("Testing Audio Streaming")
    
    uri = "ws://localhost:8000/api/v1/ws/audio/ingest?session_id=integration_stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for connection
            msg = await websocket.recv()
            data = json.loads(msg)
            assert data["type"] == "connected"
            print_success("Connected to WebSocket")
            
            # Test 1: Silent audio
            print_info("Testing silent audio...")
            silent_audio = np.zeros(16000, dtype=np.int16)
            await websocket.send(silent_audio.tobytes())
            
            msg = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            data = json.loads(msg)
            if data["type"] == "silence":
                print_success("Silent audio detected correctly")
            
            # Test 2: Simple tone
            print_info("Testing simple tone (440 Hz)...")
            tone_audio = generate_test_audio(440, 1.0)
            await websocket.send(tone_audio.tobytes())
            
            msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(msg)
            
            if data["type"] == "partial":
                partial_data = data["data"]
                
                # Check spectral features
                spectral = partial_data.get("spectral", {})
                centroid = spectral.get("centroid", 0)
                rms = spectral.get("rms", 0)
                
                print_success(f"Spectral analysis: centroid={centroid:.1f} Hz, RMS={rms:.3f}")
                
                # Check emotion
                emotion = partial_data.get("emotion", {})
                print_success(f"Emotion: {emotion.get('label', 'N/A')} (confidence: {emotion.get('confidence', 0):.2f})")
                
                # Check hologram hint
                hologram = partial_data.get("hologram_hint", {})
                if hologram:
                    assert 0 <= hologram["hue"] <= 360
                    assert 0 <= hologram["intensity"] <= 1
                    assert 0 <= hologram["psi"] <= 1
                    print_success(f"Hologram hint: hue={hologram['hue']:.1f}°, intensity={hologram['intensity']:.2f}, psi={hologram['psi']:.2f}")
            
            # Test 3: Complex audio
            print_info("Testing complex audio...")
            complex_audio = generate_complex_audio(1.0)
            await websocket.send(complex_audio.tobytes())
            
            msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(msg)
            
            if data["type"] == "partial":
                print_success("Complex audio processed")
                
                # Check if transcript exists (might be empty for test audio)
                transcript = data["data"].get("transcript", "")
                if transcript:
                    print_info(f"Transcript: '{transcript[:50]}...'")
            
            # Test 4: Finalize stream
            print_info("Finalizing stream...")
            await websocket.send(json.dumps({"type": "finalize"}))
            
            msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(msg)
            
            if data["type"] == "final":
                print_success("Stream finalized successfully")
                
                # Check final metrics if available
                final_data = data.get("data", {})
                metrics = final_data.get("metrics", {})
                if metrics:
                    print_info(f"Total chunks: {metrics.get('total_chunks', 0)}")
                    print_info(f"Processing time: {metrics.get('processing_time', 0):.2f} s")
                    print_info(f"Real-time factor: {metrics.get('real_time_factor', 0):.2f}")
            
            return True
            
    except Exception as e:
        print_error(f"Streaming test failed: {str(e)}")
        return False

async def test_concurrent_connections() -> bool:
    """Test multiple concurrent connections"""
    print_header("Testing Concurrent Connections")
    
    async def client_session(session_id: str, frequency: float) -> bool:
        uri = f"ws://localhost:8000/api/v1/ws/audio/ingest?session_id={session_id}"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Connect
                msg = await websocket.recv()
                assert json.loads(msg)["type"] == "connected"
                
                # Send audio
                audio = generate_test_audio(frequency, 0.5)
                await websocket.send(audio.tobytes())
                
                # Get result
                msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(msg)
                
                if data["type"] == "partial":
                    centroid = data["data"]["spectral"]["centroid"]
                    print_success(f"{session_id}: Processed audio at {frequency} Hz (detected: {centroid:.1f} Hz)")
                    return True
                    
        except Exception as e:
            print_error(f"{session_id}: Failed - {str(e)}")
            return False
        
        return False
    
    # Run 3 concurrent sessions
    results = await asyncio.gather(
        client_session("concurrent_1", 440),
        client_session("concurrent_2", 880),
        client_session("concurrent_3", 1320),
        return_exceptions=True
    )
    
    successful = sum(1 for r in results if r is True)
    print_info(f"Successful concurrent connections: {successful}/3")
    
    return successful == 3

async def test_error_handling() -> bool:
    """Test error handling and recovery"""
    print_header("Testing Error Handling")
    
    uri = "ws://localhost:8000/api/v1/ws/audio/ingest?session_id=integration_errors"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Connect
            await websocket.recv()
            
            # Test 1: Invalid control message
            print_info("Testing invalid control message...")
            await websocket.send(json.dumps({"type": "invalid_command"}))
            
            # Should still work after error
            audio = generate_test_audio(440, 0.5)
            await websocket.send(audio.tobytes())
            
            msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(msg)
            
            if data["type"] == "partial":
                print_success("Recovery after invalid message successful")
            
            # Test 2: Malformed audio data
            print_info("Testing malformed audio...")
            await websocket.send(b"not_audio_data")
            
            # Should handle gracefully
            await asyncio.sleep(0.5)
            
            # Test 3: Empty audio
            print_info("Testing empty audio...")
            await websocket.send(b"")
            
            # Should still be connected
            await websocket.send(json.dumps({"type": "ping"}))
            msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(msg)
            
            if data["type"] == "pong":
                print_success("Connection maintained after errors")
                return True
                
    except Exception as e:
        print_error(f"Error handling test failed: {str(e)}")
        return False
    
    return False

async def test_performance() -> bool:
    """Test streaming performance"""
    print_header("Testing Performance")
    
    uri = "ws://localhost:8000/api/v1/ws/audio/ingest?session_id=integration_performance"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Connect
            await websocket.recv()
            
            # Stream for 5 seconds
            print_info("Streaming audio for 5 seconds...")
            start_time = time.time()
            chunks_sent = 0
            bytes_sent = 0
            responses_received = 0
            
            while time.time() - start_time < 5.0:
                # Generate 100ms chunks
                audio = generate_test_audio(440 + chunks_sent * 10, 0.1)
                await websocket.send(audio.tobytes())
                
                chunks_sent += 1
                bytes_sent += len(audio.tobytes())
                
                # Non-blocking receive
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    data = json.loads(msg)
                    if data["type"] == "partial":
                        responses_received += 1
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(0.1)
            
            elapsed = time.time() - start_time
            throughput = bytes_sent / elapsed / 1024 / 1024  # MB/s
            
            print_success(f"Chunks sent: {chunks_sent}")
            print_success(f"Responses received: {responses_received}")
            print_success(f"Throughput: {throughput:.2f} MB/s")
            print_success(f"Average latency: {(elapsed / responses_received * 1000):.1f} ms" if responses_received > 0 else "N/A")
            
            # Request final stats
            await websocket.send(json.dumps({"type": "stats"}))
            
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(msg)
                
                if data["type"] == "stats":
                    stats = data.get("data", {})
                    print_info(f"Server stats - Bytes: {stats.get('bytes_received', 0)}, Chunks: {stats.get('chunks_processed', 0)}")
            except asyncio.TimeoutError:
                pass
            
            return throughput > 0.1  # At least 0.1 MB/s
            
    except Exception as e:
        print_error(f"Performance test failed: {str(e)}")
        return False

async def run_all_tests() -> bool:
    """Run all integration tests"""
    print_header("🧪 TORI WebSocket Integration Tests")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check server health first
    if not await check_server_health():
        print_error("\nServer is not healthy or not running!")
        print_info("Start the server with: uvicorn tori_backend.main:app --reload")
        return False
    
    # Run all tests
    tests = [
        ("Basic Connection", test_websocket_connection),
        ("Audio Streaming", test_audio_streaming),
        ("Concurrent Connections", test_concurrent_connections),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n" + "="*60)
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print_header("Test Summary")
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        color = Colors.OKGREEN if result else Colors.FAIL
        print(f"{color}{test_name}: {status}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Total: {passed_tests}/{total_tests} tests passed{Colors.ENDC}")
    
    if passed_tests == total_tests:
        print_success("\n🎉 All tests passed!")
        return True
    else:
        print_error(f"\n❌ {total_tests - passed_tests} tests failed")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TORI WebSocket Integration Tests")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set global host/port if needed
    # (You could modify the tests to use these)
    
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_warning("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()