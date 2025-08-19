# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
FastAPI microservice for ALAN simulation.

This provides API endpoints for running and monitoring ALAN simulations,
including streaming real-time state data for visualization in the dashboard.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alan_backend.core.oscillator.banksy_oscillator import BanksyOscillator, BanksyConfig
from alan_backend.core.controller.trs_ode import TRSController, TRSConfig
from alan_backend.core.memory.spin_hopfield import SpinHopfieldMemory, HopfieldConfig
from alan_backend.core.banksy_fusion import BanksyFusion, BanksyFusionConfig, BanksyReasoner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ALAN Simulation API",
    description="API for running and monitoring ALAN simulations",
    version="0.9.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API requests and responses
class SimulationConfig(BaseModel):
    """Configuration for a simulation run."""
    n_oscillators: int = Field(32, description="Number of oscillators")
    run_steps: int = Field(100, description="Number of steps to run")
    spin_substeps: int = Field(8, description="Spin substeps per phase step")
    coupling_type: str = Field("modular", description="Coupling type: uniform, modular, random")


class SimulationState(BaseModel):
    """Current state of a simulation."""
    step: int
    time: float
    order_parameter: float
    mean_phase: float
    n_effective: int
    active_concepts: Dict[str, float] = {}
    trs_loss: Optional[float] = None
    rollback: bool = False


class SimulationResult(BaseModel):
    """Final result of a simulation run."""
    id: str
    config: SimulationConfig
    final_state: SimulationState
    history_summary: Dict[str, List[float]]
    message: str


# Global store of running simulations
active_simulations: Dict[str, Any] = {}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "ALAN Simulation API",
        "version": "0.9.0",
        "endpoints": [
            {"path": "/simulate", "method": "POST", "description": "Start a new simulation"},
            {"path": "/simulate/{sim_id}", "method": "GET", "description": "Get simulation status"},
            {"path": "/simulate/{sim_id}/stream", "method": "GET", "description": "Stream simulation results"},
            {"path": "/ws/simulate", "method": "WebSocket", "description": "WebSocket for real-time simulation data"},
        ]
    }


@app.post("/simulate", response_model=SimulationResult)
async def create_simulation(config: SimulationConfig, background_tasks: BackgroundTasks):
    """Start a new simulation with the given configuration."""
    # Generate a unique ID for this simulation
    import uuid
    sim_id = str(uuid.uuid4())
    
    # Create the simulation components based on config
    n_oscillators = config.n_oscillators
    
    # Create oscillator with appropriate coupling
    oscillator_config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
    oscillator = BanksyOscillator(n_oscillators, oscillator_config)
    
    # Set up coupling matrix based on type
    coupling = None
    if config.coupling_type == "uniform":
        coupling = np.ones((n_oscillators, n_oscillators)) * 0.1
        np.fill_diagonal(coupling, 0.0)
    elif config.coupling_type == "modular":
        coupling = np.ones((n_oscillators, n_oscillators)) * 0.05
        np.fill_diagonal(coupling, 0.0)
        
        # Create two modules with stronger internal coupling
        module_size = n_oscillators // 2
        for i in range(module_size):
            for j in range(module_size):
                if i != j:
                    coupling[i, j] = 0.3
                    coupling[i + module_size, j + module_size] = 0.3
    elif config.coupling_type == "random":
        import numpy as np
        np.random.seed(42)  # For reproducibility
        coupling = np.random.uniform(0, 0.2, (n_oscillators, n_oscillators))
        np.fill_diagonal(coupling, 0.0)
    
    if coupling is not None:
        oscillator.set_coupling(coupling)
    
    # Create the simulator
    fusion_config = BanksyFusionConfig(
        oscillator=oscillator_config,
        controller=TRSConfig(dt=0.01, train_steps=20),
        memory=HopfieldConfig(beta=1.5, max_iterations=50),
    )
    
    # Simple concept names for visualization
    concept_labels = [f"concept_{i}" for i in range(n_oscillators)]
    
    # Initialize simulator
    simulator = BanksyFusion(n_oscillators, fusion_config, concept_labels)
    
    # Store in active simulations
    active_simulations[sim_id] = {
        "config": config,
        "simulator": simulator,
        "step": 0,
        "running": False,
        "states": [],
    }
    
    # Run simulation in background
    background_tasks.add_task(run_simulation, sim_id, config)
    
    # Return initial response
    return SimulationResult(
        id=sim_id,
        config=config,
        final_state=SimulationState(
            step=0,
            time=0.0,
            order_parameter=0.0,
            mean_phase=0.0,
            n_effective=0,
        ),
        history_summary={},
        message="Simulation started in background",
    )


async def run_simulation(sim_id: str, config: SimulationConfig):
    """Run a simulation in the background."""
    sim_data = active_simulations[sim_id]
    simulator = sim_data["simulator"]
    
    # Mark as running
    sim_data["running"] = True
    
    # Run the specified number of steps
    for _ in range(config.run_steps):
        metrics = simulator.step()
        
        # Store metrics
        state = SimulationState(
            step=metrics["step"],
            time=metrics["step"] * simulator.config.oscillator.dt,
            order_parameter=metrics["order_parameter"],
            mean_phase=metrics["mean_phase"],
            n_effective=metrics["n_effective"],
            active_concepts=metrics["active_concepts"],
            trs_loss=metrics.get("trs_loss"),
            rollback=metrics.get("rollback", False),
        )
        sim_data["states"].append(state)
        sim_data["step"] = metrics["step"]
        
        # Slow down simulation for visualization
        await asyncio.sleep(0.1)
    
    # Mark as finished
    sim_data["running"] = False


@app.get("/simulate/{sim_id}", response_model=SimulationResult)
async def get_simulation(sim_id: str):
    """Get the current state of a simulation."""
    if sim_id not in active_simulations:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    sim_data = active_simulations[sim_id]
    
    # Get the latest state
    states = sim_data["states"]
    final_state = states[-1] if states else SimulationState(
        step=0, time=0.0, order_parameter=0.0, mean_phase=0.0, n_effective=0
    )
    
    # Prepare history summary
    history_summary = {
        "order_parameter": [s.order_parameter for s in states],
        "n_effective": [s.n_effective for s in states],
    }
    
    # Return result
    return SimulationResult(
        id=sim_id,
        config=sim_data["config"],
        final_state=final_state,
        history_summary=history_summary,
        message="Simulation running" if sim_data["running"] else "Simulation complete",
    )


@app.get("/simulate/{sim_id}/stream")
async def stream_simulation(sim_id: str):
    """Stream simulation results as a server-sent events stream."""
    if sim_id not in active_simulations:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    async def event_generator():
        """Generate SSE events for the simulation."""
        sim_data = active_simulations[sim_id]
        last_state_idx = 0
        
        while sim_data["running"] or last_state_idx < len(sim_data["states"]):
            # Check for new states
            if last_state_idx < len(sim_data["states"]):
                state = sim_data["states"][last_state_idx]
                yield f"data: {json.dumps(state.dict())}\n\n"
                last_state_idx += 1
            
            # Sleep before checking again
            await asyncio.sleep(0.2)
        
        # Send completion message
        yield f"data: {json.dumps({'event': 'complete'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.websocket("/ws/simulate")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation data."""
    await websocket.accept()
    
    # Create a task for the simulation
    sim_task = None
    
    try:
        # Wait for initial configuration message
        config_data = await websocket.receive_json()
        config = SimulationConfig(**config_data)
        
        # Create a simulation
        sim_id = str(uuid.uuid4())
        
        # Same setup as in create_simulation
        n_oscillators = config.n_oscillators
        oscillator_config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
        oscillator = BanksyOscillator(n_oscillators, oscillator_config)
        
        # Set up coupling matrix based on type
        # ... (same code as in create_simulation)
        
        # Initialize simulator
        fusion_config = BanksyFusionConfig(
            oscillator=oscillator_config,
            controller=TRSConfig(dt=0.01, train_steps=20),
            memory=HopfieldConfig(beta=1.5, max_iterations=50),
        )
        
        concept_labels = [f"concept_{i}" for i in range(n_oscillators)]
        simulator = BanksyFusion(n_oscillators, fusion_config, concept_labels)
        
        # Define the simulation coroutine
        async def run_simulation():
            try:
                # Run simulation and stream results
                for _ in range(config.run_steps):
                    metrics = simulator.step()
                    
                    # Create state object
                    state = SimulationState(
                        step=metrics["step"],
                        time=metrics["step"] * simulator.config.oscillator.dt,
                        order_parameter=metrics["order_parameter"],
                        mean_phase=metrics["mean_phase"],
                        n_effective=metrics["n_effective"],
                        active_concepts=metrics["active_concepts"],
                        trs_loss=metrics.get("trs_loss"),
                        rollback=metrics.get("rollback", False),
                    )
                    
                    # Send state to client with timeout to prevent back-pressure
                    try:
                        await asyncio.wait_for(
                            websocket.send_json(state.dict(), mode="text"),
                            timeout=0.5  # 500ms timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket send timeout - client too slow, dropping frame")
                        # Skip a few frames to catch up
                        await asyncio.sleep(0.3)
                        continue
                    except websocket.exceptions.ConnectionClosed:
                        # Connection closed by client, exit quietly
                        logger.info("WebSocket connection closed by client")
                        return
                    
                    # Slow down for visualization
                    await asyncio.sleep(0.1)
                
                # Send completion message
                try:
                    await websocket.send_json({"event": "complete"})
                except websocket.exceptions.ConnectionClosed:
                    # Silently exit if connection already closed
                    pass
            except asyncio.CancelledError:
                # Task was cancelled, exit cleanly
                logger.info("Simulation task cancelled")
                raise
            except Exception as e:
                # Log error but don't crash the task
                logger.error(f"Simulation error: {e}")
                try:
                    await websocket.send_json({"error": str(e)})
                except websocket.exceptions.ConnectionClosed:
                    # Connection already closed, can't send error
                    pass
        
        # Create and start the simulation task
        sim_task = asyncio.create_task(run_simulation())
        
        # Wait for the simulation to complete or for a disconnection
        await sim_task
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected - cleaning up resources")
        # Cancel the simulation task if it's running
        if sim_task and not sim_task.done():
            sim_task.cancel()
            try:
                # Wait for the task to actually complete cancellation
                await asyncio.wait([sim_task])
                logger.info("Simulation task successfully cancelled and resources released")
            except Exception as e:
                logger.error(f"Error during task cancellation: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            # If we can't send the error, just log it
            pass
        
        # Cancel the simulation task if it's running
        if sim_task and not sim_task.done():
            sim_task.cancel()
            try:
                await sim_task
            except asyncio.CancelledError:
                # This is expected when we cancel the task
                pass


# Testing the API
if __name__ == "__main__":
    import uvicorn
    import numpy as np  # Needed for coupling matrix setup
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
