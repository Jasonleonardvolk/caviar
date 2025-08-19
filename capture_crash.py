#!/usr/bin/env python3
"""
🔍 CRASH CAPTURE SCRIPT - Capture extraction crashes with full logging
Records all output, errors, and crash details for debugging
"""

import subprocess
import sys
import os
import time
import signal
from datetime import datetime
from pathlib import Path

class CrashCapture:
    def __init__(self):
        self.log_dir = Path(__file__).parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"api_crash_{timestamp}.log"
        self.error_file = self.log_dir / f"api_errors_{timestamp}.log"
        
    def log(self, message, is_error=False):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        
        print(formatted)
        
        log_file = self.error_file if is_error else self.log_file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted + '\n')
            f.flush()
    
    def run_server_with_capture(self):
        """Run the API server with full crash capture"""
        
        self.log("🔍 CRASH CAPTURE STARTING")
        self.log("=" * 50)
        self.log(f"📝 Logs will be saved to: {self.log_file}")
        self.log(f"❌ Errors will be saved to: {self.error_file}")
        
        try:
            # Start the dynamic API server process
            self.log("🚀 Starting dynamic API server with crash capture...")
            
            # Use the dynamic server script
            cmd = [sys.executable, "start_dynamic_api.py"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            self.log(f"📍 Server process started with PID: {process.pid}")
            
            # Monitor both stdout and stderr in real-time
            import threading
            
            def monitor_stdout():
                """Monitor and log stdout"""
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    self.log(f"[STDOUT] {line.strip()}")
            
            def monitor_stderr():
                """Monitor and log stderr"""
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    self.log(f"[STDERR] {line.strip()}", is_error=True)
            
            # Start monitoring threads
            stdout_thread = threading.Thread(target=monitor_stdout, daemon=True)
            stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete or crash
            return_code = process.wait()
            
            self.log(f"🛑 Server process ended with return code: {return_code}")
            
            if return_code != 0:
                self.log("❌ SERVER CRASHED!", is_error=True)
                self.log(f"❌ Return code: {return_code}", is_error=True)
                
                # Get any remaining output
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    self.log(f"[FINAL STDOUT] {remaining_stdout}", is_error=True)
                if remaining_stderr:
                    self.log(f"[FINAL STDERR] {remaining_stderr}", is_error=True)
            else:
                self.log("✅ Server ended gracefully")
                
        except KeyboardInterrupt:
            self.log("🛑 Interrupted by user (Ctrl+C)")
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
                
        except Exception as e:
            self.log(f"❌ CRASH CAPTURE ERROR: {e}", is_error=True)
            import traceback
            self.log(f"❌ Traceback: {traceback.format_exc()}", is_error=True)
        
        finally:
            self.log("🔍 CRASH CAPTURE COMPLETE")
            self.log(f"📝 Check logs: {self.log_file}")
            self.log(f"❌ Check errors: {self.error_file}")

def main():
    """Main crash capture function"""
    print("🔍 TORI API CRASH CAPTURE")
    print("=" * 40)
    print("This will run the API server and capture all crash details")
    print("Press Ctrl+C to stop")
    print()
    
    capture = CrashCapture()
    capture.run_server_with_capture()

if __name__ == "__main__":
    main()
