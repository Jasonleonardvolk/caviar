"""
MCP 2.0 PCC State Broadcasting Test

This script tests the MCP 2.0 server by:
1. Establishing multiple WebSocket connections
2. Sending PCC state packets
3. Verifying broadcast and latency
"""

import sys
import json
import asyncio
import signal
import argparse
import websockets
import httpx
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from websockets.exceptions import ConnectionClosed

# Default URLs
MCP_BASE_URL = "127.0.0.1:8787"
MCP_HTTP_URL = f"http://{MCP_BASE_URL}/pcc_state"
MCP_WS_URL = f"ws://{MCP_BASE_URL}/ws"

# Test configuration
MAX_LATENCY_MS = 50  # Maximum acceptable latency in milliseconds


class WebSocketClient:
    """WebSocket client for testing MCP server broadcast."""
    
    def __init__(self, url, name="Client"):
        self.url = url
        self.name = name
        self.received_messages = []
        self.connected = False
        self.websocket = None
        self.receive_task = None
        self.start_time = None
        self.latencies = []
    
    async def connect(self):
        """Connect to WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            self.start_time = time.time()
            print(f"✅ {self.name} connected to {self.url}")
            return True
        except Exception as e:
            print(f"❌ {self.name} failed to connect: {e}")
            return False
    
    async def start_receiving(self):
        """Start receiving messages in background."""
        self.receive_task = asyncio.create_task(self.receive_messages())
        return self.receive_task
    
    async def receive_messages(self):
        """Receive and store messages."""
        if not self.websocket:
            return
            
        try:
            while self.connected:
                message = await self.websocket.recv()
                recv_time = time.time()
                data = json.loads(message)
                
                # Store received message
                self.received_messages.append({
                    "data": data,
                    "time": recv_time
                })
                
                # Calculate latency if message has timestamp
                if "timestamp" in data:
                    sent_time = data["timestamp"]
                    latency = (recv_time - sent_time) * 1000  # ms
                    self.latencies.append(latency)
                    
                print(f"📥 {self.name} received message: step={data.get('step', 'N/A')}")
                
        except ConnectionClosed:
            print(f"🔌 {self.name} connection closed")
            self.connected = False
        except Exception as e:
            print(f"❌ {self.name} error: {e}")
            self.connected = False
    
    async def close(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            if self.receive_task:
                self.receive_task.cancel()
            print(f"🔌 {self.name} disconnected")
            

async def test_multiple_clients():
    """Test broadcasting to multiple WebSocket clients."""
    print("\n📡 Testing broadcast to multiple WebSocket clients\n")
    
    # Create clients
    client1 = WebSocketClient(MCP_WS_URL, "Client 1")
    client2 = WebSocketClient(MCP_WS_URL, "Client 2")
    
    # Connect clients
    if not await client1.connect() or not await client2.connect():
        print("❌ Failed to connect all clients")
        return False
    
    # Start receiving in background
    task1 = await client1.start_receiving()
    task2 = await client2.start_receiving()
    
    # Wait a moment to establish connections
    await asyncio.sleep(0.5)
    
    # Generate test packet
    test_data = {
        "step": 1,
        "phases": np.linspace(0, 2 * np.pi, 32).tolist(),
        "spins": [1 if i % 2 == 0 else -1 for i in range(32)],
        "energy": -0.5,
        "timestamp": time.time()  # Add timestamp for latency calculation
    }
    
    print(f"\nSending test PCC state to {MCP_HTTP_URL}")
    
    try:
        # Send to MCP server
        send_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MCP_HTTP_URL,
                json=test_data,
                timeout=5.0
            )
        
        # Check response
        if response.status_code != 200:
            print(f"❌ Server returned status {response.status_code}")
            return False
            
        # Wait to receive messages
        await asyncio.sleep(0.5)
        
        # Check if both clients received the message
        if len(client1.received_messages) == 0 or len(client2.received_messages) == 0:
            print("❌ Not all clients received the broadcast")
            return False
            
        # Check latency between clients
        recv_time1 = client1.received_messages[0]["time"]
        recv_time2 = client2.received_messages[0]["time"]
        latency_between_clients = abs(recv_time1 - recv_time2) * 1000  # ms
        
        print(f"\nLatency between clients: {latency_between_clients:.2f} ms")
        print(f"Server-to-client latency: {(recv_time1 - send_time) * 1000:.2f} ms (Client 1)")
        
        # Close connections
        await client1.close()
        await client2.close()
        
        # Test result
        if latency_between_clients <= MAX_LATENCY_MS:
            print(f"\n✅ Broadcast test PASSED - Both clients received within {latency_between_clients:.2f} ms")
            return True
        else:
            print(f"\n❌ Broadcast test FAILED - Latency too high: {latency_between_clients:.2f} ms > {MAX_LATENCY_MS} ms")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        await client1.close()
        await client2.close()
        return False


async def test_throughput(rate=100, duration=5):
    """Test server throughput at specified message rate."""
    print(f"\n🚀 Testing throughput at {rate} messages/sec for {duration} seconds\n")
    
    # Calculate total messages and interval
    total_messages = rate * duration
    interval = 1.0 / rate
    
    # Connect to WebSocket to receive messages
    client = WebSocketClient(MCP_WS_URL, "Throughput Monitor")
    if not await client.connect():
        return False
        
    await client.start_receiving()
    
    # Prepare test data
    test_data_template = {
        "phases": np.linspace(0, 2 * np.pi, 32).tolist(),
        "spins": [1 if i % 2 == 0 else -1 for i in range(32)],
        "energy": -0.5
    }
    
    success_count = 0
    failure_count = 0
    latencies = []
    
    print(f"Sending {total_messages} messages at {rate} messages/sec...")
    
    async with httpx.AsyncClient() as http_client:
        start_time = time.time()
        
        for i in range(total_messages):
            send_time = time.time()
            test_data = test_data_template.copy()
            test_data["step"] = i
            test_data["timestamp"] = send_time
            
            try:
                response = await http_client.post(
                    MCP_HTTP_URL,
                    json=test_data,
                    timeout=1.0
                )
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    failure_count += 1
                    
                # Calculate and control rate
                elapsed = time.time() - send_time
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
                    
            except Exception as e:
                failure_count += 1
                if failure_count % 10 == 0:
                    print(f"Error sending message {i}: {e}")
        
        # Wait for all messages to be received
        await asyncio.sleep(1.0)
        
        # Final statistics
        elapsed_time = time.time() - start_time
        actual_rate = success_count / elapsed_time
        
        # Calculate latencies
        for msg in client.received_messages:
            if "timestamp" in msg["data"]:
                latency = (msg["time"] - msg["data"]["timestamp"]) * 1000  # ms
                latencies.append(latency)
                
        # Close client
        await client.close()
        
        # Print results
        print(f"\nThroughput Results:")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Target rate: {rate} messages/sec")
        print(f"Actual rate: {actual_rate:.2f} messages/sec")
        print(f"Messages sent: {success_count + failure_count}")
        print(f"Success rate: {success_count/(success_count+failure_count)*100:.2f}%")
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else float('inf')
            print(f"Average latency: {avg_latency:.2f} ms")
            print(f"95th percentile latency: {p95_latency:.2f} ms")
            
            if p95_latency <= MAX_LATENCY_MS:
                print(f"\n✅ Throughput test PASSED - p95 latency: {p95_latency:.2f} ms ≤ {MAX_LATENCY_MS} ms")
                return True
            else:
                print(f"\n❌ Throughput test FAILED - p95 latency too high: {p95_latency:.2f} ms > {MAX_LATENCY_MS} ms")
                print("Consider enabling uvloop and httptools for better performance")
                return False
        else:
            print("\n❌ Throughput test FAILED - No latency data collected")
            return False


def test_pcc_broadcast():
    """Test sending a PCC state to the MCP server."""
    # Generate a test packet
    test_data = {
        "step": 1,
        "phases": np.linspace(0, 2 * np.pi, 32).tolist(),
        "spins": [1 if i % 2 == 0 else -1 for i in range(32)],
        "energy": -0.5
    }
    
    print(f"Sending test PCC state to {MCP_HTTP_URL}")
    print(f"Data: step={test_data['step']}, energy={test_data['energy']}")
    print(f"Phases: {len(test_data['phases'])} values")
    print(f"Spins: {len(test_data['spins'])} values")
    
    try:
        # Send to MCP server
        start_time = time.time()
        response = httpx.post(
            MCP_HTTP_URL,
            json=test_data,
            timeout=5.0
        )
        elapsed = (time.time() - start_time) * 1000
        
        # Check response
        print(f"\nResponse from server ({elapsed:.2f}ms):")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            print("\nBasic HTTP test PASSED ✅")
            return True
        else:
            print("\nBasic HTTP test FAILED ❌")
            return False
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nBasic HTTP test FAILED ❌")
        print("\nMake sure the MCP server is running:")
        print("  - Run start-mcp-server.bat")
        print("  - Check if port 8787 is already in use")
        return False


def check_server_health():
    """Check if the MCP server is running."""
    try:
        response = httpx.get(f"http://{MCP_BASE_URL}/health", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            print(f"Server is healthy ✅")
            print(f"Wire Format: {data.get('wire_format', 'unknown')}")
            print(f"Connected WebSocket clients: {data.get('websocket_clients', 0)}")
            return True
        else:
            print(f"Server returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking server health: {e}")
        return False


def check_server_metrics():
    """Check server metrics."""
    try:
        response = httpx.get(f"http://{MCP_BASE_URL}/metrics", timeout=2.0)
        if response.status_code == 200:
            metrics = response.json()
            print("\nServer Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            return True
        else:
            print(f"Error fetching metrics: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking metrics: {e}")
        return False


async def run_tests(args):
    """Run all specified tests."""
    results = {}
    
    # Basic broadcast test
    if args.basic or args.all:
        results["basic"] = test_pcc_broadcast()
    
    # WebSocket clients test
    if args.ws or args.all:
        results["ws"] = await test_multiple_clients()
    
    # Throughput test
    if args.throughput or args.all:
        results["throughput"] = await test_throughput(rate=args.rate, duration=args.duration)
    
    # Check metrics after tests if available
    if any(results.values()):
        check_server_metrics()
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test, passed in results.items():
        print(f"{test.capitalize(): <15}: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    # Return overall result
    return all(results.values())


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MCP 2.0 Server Test Suite")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--basic", action="store_true", help="Run basic HTTP test")
    parser.add_argument("--ws", action="store_true", help="Run WebSocket clients test")
    parser.add_argument("--throughput", action="store_true", help="Run throughput test")
    parser.add_argument("--rate", type=int, default=100, help="Messages per second for throughput test")
    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds for throughput test")
    
    args = parser.parse_args()
    
    # If no tests specified, default to all
    if not (args.all or args.basic or args.ws or args.throughput):
        args.all = True
        
    return args


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    # Parse arguments
    args = parse_arguments()
    
    # First check if server is running
    print("Checking MCP server health...\n")
    if not check_server_health():
        print("\nMCP server is not responding. Please start it first:")
        print("  - Run start-mcp-server.bat")
        sys.exit(1)
    
    print("\n-----------------------------------\n")
    
    # Run tests
    result = asyncio.run(run_tests(args))
    
    # Exit with status code
    sys.exit(0 if result else 1)
