"""
ELFIN Debug System Standalone Demo

This demo runs without depending on the full ELFIN parser to demonstrate the debugging features.
"""

import time
import math
import random
import argparse
import threading
import numpy as np
from threading import Thread

# Import debug components directly
# You can comment these imports and use the local classes if the imports don't work
try:
    from alan_backend.elfin.debug.lyapunov_monitor import lyapunov_monitor
except ImportError:
    # Fallback implementation for demo purposes
    import json
    import asyncio
    import logging
    import websockets
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("elfin.debug.standalone")
    
    class LyapunovMonitor:
        """Simplified LyapunovMonitor implementation for the standalone demo."""
        
        def __init__(self, port=8642):
            self.port = port
            self.running = False
            self.clients = set()
            self.server = None
            self.server_task = None
            self.seq_counter = 0
            
            self.barrier_functions = {}
            self.breakpoints = []
            self.state_vars = {}
            self.events = []
            self.V = 0.0
            self.Vdot = 0.0
            self.metadata = {"mode": "default", "status": "initializing", "units": {}}
            self.last_send_time = 0
            self.min_update_interval = 0.01
            self.handshake_packet = {
                "type": "handshake",
                "schema": {"vars": {}, "barriers": []},
                "dt_nominal": 0.01
            }
        
        async def start_server(self):
            if self.server is not None:
                logger.warning("Server already running")
                return
            
            self.running = True
            self.server = await websockets.serve(
                self.client_handler, "localhost", self.port
            )
            logger.info(f"Lyapunov monitor server started on port {self.port}")
        
        def start(self):
            def run_server_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.server_task = loop.create_task(self.start_server())
                loop.run_forever()
            
            server_thread = threading.Thread(target=run_server_thread, daemon=True)
            server_thread.start()
            logger.info("Lyapunov monitor started in background thread")
            
            self.handshake_packet["schema"]["barriers"] = list(self.barrier_functions.keys())
        
        async def stop_server(self):
            if self.server is None:
                return
            
            self.running = False
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            
            close_coroutines = [client.close() for client in self.clients]
            if close_coroutines:
                await asyncio.gather(*close_coroutines, return_exceptions=True)
            
            self.clients.clear()
            logger.info("Lyapunov monitor server stopped")
        
        def stop(self):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.stop_server())
            else:
                loop.run_until_complete(self.stop_server())
            logger.info("Lyapunov monitor stopped")
        
        async def client_handler(self, websocket, path):
            try:
                self.clients.add(websocket)
                logger.info(f"Client connected: {websocket.remote_address}")
                
                await websocket.send(json.dumps(self.handshake_packet))
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.handle_client_message(websocket, data)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON message from client")
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client disconnected: {websocket.remote_address}")
            finally:
                self.clients.remove(websocket)
        
        async def handle_client_message(self, websocket, data):
            # Future implementation
            pass
        
        async def broadcast(self, message):
            if not self.clients:
                return
            
            if isinstance(message, dict):
                message = json.dumps(message)
            
            websockets_to_remove = set()
            
            for websocket in self.clients:
                try:
                    await websocket.send(message)
                except websockets.exceptions.ConnectionClosed:
                    websockets_to_remove.add(websocket)
            
            for websocket in websockets_to_remove:
                self.clients.remove(websocket)
        
        def register_barrier_function(self, name, func, threshold, description=""):
            self.barrier_functions[name] = {
                "func": func,
                "threshold": threshold,
                "description": description,
                "value": None
            }
            
            self.handshake_packet["schema"]["barriers"] = list(self.barrier_functions.keys())
            
            logger.info(f"Registered barrier function: {name}")
        
        def register_state_var(self, name, unit=None):
            self.state_vars[name] = None
            
            if unit:
                if "vars" not in self.handshake_packet["schema"]:
                    self.handshake_packet["schema"]["vars"] = {}
                self.handshake_packet["schema"]["vars"][name] = unit
                
                self.metadata["units"][name] = unit
            
            logger.info(f"Registered state variable: {name} ({unit or 'no unit'})")
        
        def set_lyapunov_function(self, V_func, Vdot_func):
            self.V_func = V_func
            self.Vdot_func = Vdot_func
            logger.info("Lyapunov functions set")
        
        def add_breakpoint(self, condition, expression):
            self.breakpoints.append({
                "condition": condition,
                "expression": expression,
                "enabled": True
            })
            logger.info(f"Added breakpoint: {condition}")
        
        def update(self, **kwargs):
            # Update state variables
            for name, value in kwargs.items():
                self.state_vars[name] = value
            
            # Update Lyapunov function and derivative
            try:
                if hasattr(self, 'V_func'):
                    self.V = self.V_func(**self.state_vars)
                if hasattr(self, 'Vdot_func'):
                    self.Vdot = self.Vdot_func(**self.state_vars)
            except Exception as e:
                logger.error(f"Error computing Lyapunov functions: {e}")
            
            # Update barrier functions
            for name, barrier in self.barrier_functions.items():
                try:
                    barrier["value"] = barrier["func"](**self.state_vars)
                except Exception as e:
                    logger.error(f"Error computing barrier function {name}: {e}")
            
            # Check breakpoints
            event = None
            for bp in self.breakpoints:
                if not bp["enabled"]:
                    continue
                    
                try:
                    eval_context = {**self.state_vars}
                    eval_context['V'] = self.V
                    eval_context['Vdot'] = self.Vdot
                    
                    for b_name, barrier in self.barrier_functions.items():
                        eval_context[b_name] = barrier["value"]
                    
                    if eval(bp["expression"], {"__builtins__": {}}, eval_context):
                        event = f"break:{bp['condition']}"
                        self.events.append({
                            "type": "breakpoint",
                            "condition": bp["condition"],
                            "time": time.time()
                        })
                        break
                except Exception as e:
                    logger.error(f"Error evaluating breakpoint {bp['condition']}: {e}")
            
            # Check barrier violations
            if event is None:
                for name, barrier in self.barrier_functions.items():
                    if barrier["value"] is not None and barrier["value"] < barrier["threshold"]:
                        event = f"warn:barrier:{name}"
                        self.events.append({
                            "type": "barrier_violation",
                            "barrier": name,
                            "value": barrier["value"],
                            "threshold": barrier["threshold"],
                            "time": time.time()
                        })
                        break
            
            # Check Lyapunov instability
            if event is None and self.Vdot > 0:
                event = f"warn:unstable"
                self.events.append({
                    "type": "instability",
                    "V": self.V,
                    "Vdot": self.Vdot,
                    "time": time.time()
                })
            
            # Send update to connected clients
            self.send_update(event)
        
        def send_update(self, event=None):
            now = time.time()
            if now - self.last_send_time < self.min_update_interval and event is None:
                return
            
            self.last_send_time = now
            
            packet = {
                "type": "state",
                "seq": self.seq_counter,
                "t": now,
                "vars": {**self.state_vars},
                "V": self.V,
                "Vdot": self.Vdot,
                "barriers": {
                    name: barrier["value"] 
                    for name, barrier in self.barrier_functions.items()
                    if barrier["value"] is not None
                },
                "event": event,
                "meta": self.metadata
            }
            
            self.seq_counter += 1
            
            asyncio.get_event_loop().create_task(self.broadcast(packet))
    
    # Create singleton instance
    lyapunov_monitor = LyapunovMonitor()


# Simple pendulum example
class Pendulum:
    """A simple pendulum system for demonstration."""
    
    def __init__(self, mass=1.0, length=1.0, damping=0.1, g=9.81):
        """
        Initialize the pendulum.
        
        Args:
            mass: Pendulum mass (kg)
            length: Pendulum length (m)
            damping: Damping coefficient
            g: Gravitational acceleration (m/s^2)
        """
        self.mass = mass
        self.length = length
        self.damping = damping
        self.g = g
        
        # State: [theta, omega]
        self.state = np.array([0.0, 0.0])
        
        # Control input
        self.torque = 0.0
        
        # Register with Lyapunov monitor
        self._register_with_monitor()
    
    def _register_with_monitor(self):
        """Register with the Lyapunov monitor."""
        # Register state variables with units
        lyapunov_monitor.register_state_var("theta", "rad")
        lyapunov_monitor.register_state_var("omega", "rad/s")
        lyapunov_monitor.register_state_var("energy", "J")
        
        # Set Lyapunov function (energy-based)
        lyapunov_monitor.set_lyapunov_function(
            V_func=self.energy,
            Vdot_func=self.energy_derivative
        )
        
        # Register barrier function for angle limits
        lyapunov_monitor.register_barrier_function(
            name="angle_limit",
            func=self.angle_barrier,
            threshold=0.0,
            description="Pendulum angle limit"
        )
        
        # Add breakpoints
        lyapunov_monitor.add_breakpoint(
            condition="V > 10.0",
            expression="V > 10.0"
        )
        lyapunov_monitor.add_breakpoint(
            condition="omega > 5.0",
            expression="abs(omega) > 5.0"
        )
    
    def energy(self, **kwargs):
        """
        Calculate the total energy of the pendulum (Lyapunov function).
        
        Args:
            **kwargs: State variables (ignored, using internal state)
            
        Returns:
            The energy (J)
        """
        theta, omega = self.state
        
        # Kinetic energy
        T = 0.5 * self.mass * self.length**2 * omega**2
        
        # Potential energy (zero at the bottom position)
        U = self.mass * self.g * self.length * (1 - math.cos(theta))
        
        return T + U
    
    def energy_derivative(self, **kwargs):
        """
        Calculate the time derivative of energy (Lyapunov derivative).
        
        Args:
            **kwargs: State variables (ignored, using internal state)
            
        Returns:
            The energy derivative (J/s)
        """
        _, omega = self.state
        
        # For a damped pendulum: dE/dt = -b*omega^2 + tau*omega
        return -self.damping * omega**2 + self.torque * omega
    
    def angle_barrier(self, **kwargs):
        """
        Barrier function for angle limits.
        
        Args:
            **kwargs: State variables (ignored, using internal state)
            
        Returns:
            Barrier value (positive when safe)
        """
        theta, _ = self.state
        theta_max = math.pi  # Maximum allowed angle
        
        # Barrier: B(x) = theta_max^2 - theta^2
        return theta_max**2 - theta**2
    
    def update(self, dt):
        """
        Update the pendulum state.
        
        Args:
            dt: Time step (s)
        """
        theta, omega = self.state
        
        # Pendulum dynamics
        # d(theta)/dt = omega
        # d(omega)/dt = -(g/L)*sin(theta) - (b/mL^2)*omega + tau/(mL^2)
        
        # Euler integration
        theta_dot = omega
        omega_dot = (
            -(self.g / self.length) * math.sin(theta) 
            - (self.damping / (self.mass * self.length**2)) * omega
            + self.torque / (self.mass * self.length**2)
        )
        
        theta += theta_dot * dt
        omega += omega_dot * dt
        
        # Update state
        self.state = np.array([theta, omega])
        
        # Update Lyapunov monitor with current state
        lyapunov_monitor.update(
            theta=theta,
            omega=omega,
            energy=self.energy()
        )
    
    def set_torque(self, torque):
        """Set the control torque."""
        self.torque = torque
    
    def reset(self, theta0=None, omega0=None):
        """Reset the pendulum state."""
        if theta0 is None:
            theta0 = random.uniform(-math.pi/2, math.pi/2)
        if omega0 is None:
            omega0 = random.uniform(-1.0, 1.0)
        
        self.state = np.array([theta0, omega0])
        self.torque = 0.0


def demo_controlled_pendulum():
    """Run a demo of a controlled pendulum with debug monitoring."""
    # Create pendulum
    pendulum = Pendulum(mass=1.0, length=1.0, damping=0.2)
    
    # Start Lyapunov monitor server
    try:
        lyapunov_monitor.start()
    except Exception as e:
        print(f"Error starting monitor: {e}")
        return
    
    # Initial state
    pendulum.reset(theta0=math.pi/4, omega0=0.0)
    
    # Run simulation
    dt = 0.01
    t = 0.0
    t_end = 60.0
    
    print(f"Running pendulum simulation for {t_end} seconds...")
    print("Connect to ws://localhost:8642/state to monitor")
    print("Press Ctrl+C to stop the simulation")
    
    # Destabilizing control after 20 seconds
    def destabilize_controller():
        time.sleep(20.0)
        print("Applying destabilizing control...")
        pendulum.set_torque(2.0)
        
        # Reset to stabilizing control after 10 more seconds
        time.sleep(10.0)
        print("Applying stabilizing control...")
        pendulum.set_torque(-1.0)
    
    # Start controller in a separate thread
    controller_thread = Thread(target=destabilize_controller)
    controller_thread.daemon = True
    controller_thread.start()
    
    try:
        while t < t_end:
            # Update pendulum
            pendulum.update(dt)
            
            # Sleep to simulate real-time
            time.sleep(dt)
            
            # Update time
            t += dt
            
            # Print status occasionally
            if int(t * 100) % 100 == 0:
                theta, omega = pendulum.state
                energy = pendulum.energy()
                print(f"t={t:.1f} s, θ={theta:.2f} rad, ω={omega:.2f} rad/s, E={energy:.2f} J")
    
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    
    finally:
        # Stop Lyapunov monitor
        try:
            lyapunov_monitor.stop()
        except:
            pass
        print("Simulation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELFIN Debug Standalone Demo")
    parser.add_argument("--port", type=int, default=8642, help="WebSocket server port")
    
    args = parser.parse_args()
    
    try:
        demo_controlled_pendulum()
    except Exception as e:
        print(f"Error running demo: {e}")
