"""
PCC State Sample Generator

This script generates sample PCC states and broadcasts them to the MCP server.
Used for testing the MCP WebSocket functionality and the PCC Status component.
"""

import time
import math
import random
import argparse
import numpy as np
from alan_backend.banksy.broadcast import emit_pcc


def generate_sample_pcc_state(step, num_oscillators=64):
    """
    Generate a sample PCC state with simulated phase and spin values.
    
    Args:
        step: Current simulation step
        num_oscillators: Number of oscillators to simulate
        
    Returns:
        phases, spins, energy
    """
    # Generate phases (0 to 2π) with some time-based evolution
    base_freq = 0.1
    phases = np.zeros(num_oscillators)
    
    for i in range(num_oscillators):
        # Each oscillator has a slightly different frequency
        freq = base_freq * (1 + 0.2 * (i / num_oscillators))
        phase = (step * freq) % (2 * math.pi)
        
        # Add some noise
        phase += random.uniform(-0.1, 0.1)
        if phase < 0:
            phase += 2 * math.pi
        if phase > 2 * math.pi:
            phase -= 2 * math.pi
            
        phases[i] = phase
    
    # Generate spins (-1 or 1)
    # Create clusters of similar spins with occasional flips
    spins = np.ones(num_oscillators, dtype=np.int8)
    for i in range(num_oscillators):
        if i % 8 == 0:  # Cluster boundary
            spins[i:i+8] = 1 if random.random() > 0.5 else -1
        
        # Random flips with low probability
        if random.random() < 0.02:
            spins[i] *= -1
    
    # Calculate energy (simple Ising-like energy function)
    energy = 0
    for i in range(num_oscillators):
        for j in range(i+1, num_oscillators):
            # Phase similarity contributes to energy
            phase_diff = abs(phases[i] - phases[j])
            if phase_diff > math.pi:
                phase_diff = 2 * math.pi - phase_diff
                
            # Normalize to [0, 1]
            phase_coupling = 1 - (phase_diff / math.pi)
            
            # Ising-like energy term
            energy -= phase_coupling * spins[i] * spins[j]
    
    # Normalize energy
    energy /= (num_oscillators * num_oscillators)
    
    return phases, spins, energy


def main():
    """Main function to generate and broadcast sample PCC states."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PCC State Sample Generator")
    parser.add_argument("--rate", type=float, default=2, 
                      help="Rate of PCC state updates per second (Hz)")
    parser.add_argument("--oscillators", type=int, default=64,
                      help="Number of oscillators to simulate")
    parser.add_argument("--duration", type=float, default=0,
                      help="Duration in seconds to run (0 = indefinite)")
    parser.add_argument("--quiet", action="store_true",
                      help="Reduce console output for performance testing")
    args = parser.parse_args()
    
    # Calculate sleep time between updates
    sleep_time = 1.0 / args.rate
    
    print(f"Starting PCC sample generator...")
    print(f"Rate: {args.rate} Hz (sleep: {sleep_time:.4f}s)")
    print(f"Oscillators: {args.oscillators}")
    print(f"Press Ctrl+C to stop")
    
    step = 0
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    try:
        while True:
            cycle_start = time.time()
            
            # Generate sample state
            phases, spins, energy = generate_sample_pcc_state(step, args.oscillators)
            
            # Broadcast to MCP server
            success = emit_pcc(step, phases, spins, energy)
            
            if success:
                success_count += 1
                if not args.quiet:
                    print(f"Step {step}: Broadcast successful (energy={energy:.4f})")
            else:
                fail_count += 1
                print(f"Step {step}: Broadcast failed")
            
            # Increment step
            step += 1
            
            # Check if we've reached the specified duration
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break
                
            # Calculate remaining sleep time to maintain requested rate
            elapsed = time.time() - cycle_start
            remaining = sleep_time - elapsed
            if remaining > 0:
                time.sleep(remaining)
            elif not args.quiet and elapsed > sleep_time * 1.2:
                # Warn if we're significantly behind schedule (20% over budget)
                print(f"Warning: Processing took {elapsed:.4f}s, exceeding target of {sleep_time:.4f}s")
            
    except KeyboardInterrupt:
        print("\nStopping PCC sample generator")


        # Print summary statistics
        duration = time.time() - start_time
        actual_rate = step / duration if duration > 0 else 0
        print("\nSummary:")
        print(f"Total steps: {step}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Target rate: {args.rate:.2f} Hz")
        print(f"Actual rate: {actual_rate:.2f} Hz")
        print(f"Success rate: {success_count/step*100:.2f}% ({success_count}/{step})")


if __name__ == "__main__":
    main()
