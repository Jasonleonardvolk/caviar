"""
Energy Budget Management for Dickbox
====================================

Manages and tracks energy consumption budget across services.
"""

import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from collections import defaultdict

from dickbox_config import DickboxConfig

logger = logging.getLogger(__name__)


class EnergyUnit(str, Enum):
    """Energy measurement units"""
    WATTS = "watts"
    KILOWATTS = "kilowatts"
    JOULES = "joules"
    KILOWATT_HOURS = "kWh"


@dataclass
class ServiceEnergyProfile:
    """Energy profile for a service"""
    service_name: str
    baseline_watts: float  # Baseline power consumption
    cpu_watts_per_percent: float = 0.5  # Additional watts per CPU %
    gpu_watts: float = 0.0  # GPU power if enabled
    memory_watts_per_gb: float = 3.0  # Power per GB RAM
    io_watts_per_mbps: float = 0.1  # Power per MB/s I/O
    
    def calculate_power(self, 
                       cpu_percent: float = 0,
                       memory_gb: float = 0,
                       io_mbps: float = 0,
                       gpu_active: bool = False) -> float:
        """Calculate instantaneous power consumption"""
        power = self.baseline_watts
        power += cpu_percent * self.cpu_watts_per_percent
        power += memory_gb * self.memory_watts_per_gb
        power += io_mbps * self.io_watts_per_mbps
        if gpu_active:
            power += self.gpu_watts
        return power


@dataclass
class EnergyMeasurement:
    """Point-in-time energy measurement"""
    timestamp: datetime
    service_name: str
    power_watts: float
    cpu_percent: float
    memory_gb: float
    io_mbps: float
    gpu_active: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "service_name": self.service_name,
            "power_watts": self.power_watts,
            "cpu_percent": self.cpu_percent,
            "memory_gb": self.memory_gb,
            "io_mbps": self.io_mbps,
            "gpu_active": self.gpu_active
        }


@dataclass
class EnergyBudgetState:
    """Persistent energy budget state"""
    total_budget_kwh: float  # Total energy budget in kWh
    period_start: datetime
    period_end: datetime
    consumed_kwh: float = 0.0
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    service_allocations: Dict[str, float] = field(default_factory=dict)  # Service -> kWh allocation
    warnings_sent: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_budget_kwh": self.total_budget_kwh,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "consumed_kwh": self.consumed_kwh,
            "measurements": self.measurements[-1000:],  # Keep last 1000 measurements
            "service_allocations": self.service_allocations,
            "warnings_sent": self.warnings_sent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnergyBudgetState':
        """Create from dictionary"""
        data["period_start"] = datetime.fromisoformat(data["period_start"])
        data["period_end"] = datetime.fromisoformat(data["period_end"])
        return cls(**data)


class EnergyBudget:
    """
    Manages energy consumption budget and tracking.
    
    Features:
    - Configurable budget periods (daily, weekly, monthly)
    - Per-service energy tracking
    - Predictive warnings
    - Persistent state with configurable path
    """
    
    def __init__(self, config: Optional[DickboxConfig] = None):
        self.config = config or DickboxConfig.from_env()
        self.state_path = self.config.energy_budget_path
        self.sync_interval = self.config.energy_budget_sync_interval
        
        # Service profiles
        self.service_profiles: Dict[str, ServiceEnergyProfile] = {}
        
        # Current measurements by service
        self.current_measurements: Dict[str, EnergyMeasurement] = {}
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Sync lock
        self._lock = threading.Lock()
        self._last_sync = datetime.now()
        
        # Start periodic sync
        self._sync_task = None
        
        logger.info(f"Initialized EnergyBudget with state file: {self.state_path}")
    
    def _load_state(self) -> EnergyBudgetState:
        """Load state from disk or create new"""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                state = EnergyBudgetState.from_dict(data)
                logger.info(f"Loaded energy budget state: {state.consumed_kwh:.2f}/{state.total_budget_kwh:.2f} kWh")
                return state
            except Exception as e:
                logger.error(f"Failed to load energy state: {e}")
        
        # Create default state (monthly budget)
        now = datetime.now()
        return EnergyBudgetState(
            total_budget_kwh=1000.0,  # 1 MWh per month default
            period_start=now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            period_end=(now.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        )
    
    def _save_state(self):
        """Save state to disk"""
        with self._lock:
            try:
                # Ensure directory exists
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write atomically
                temp_path = self.state_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(self.state.to_dict(), f, indent=2)
                
                # Atomic rename
                temp_path.replace(self.state_path)
                
                self._last_sync = datetime.now()
                logger.debug(f"Saved energy budget state to {self.state_path}")
                
            except Exception as e:
                logger.error(f"Failed to save energy state: {e}")
    
    async def start_sync_task(self):
        """Start periodic state sync"""
        async def sync_loop():
            while True:
                await asyncio.sleep(self.sync_interval)
                self._save_state()
        
        self._sync_task = asyncio.create_task(sync_loop())
    
    def stop_sync_task(self):
        """Stop sync task and save final state"""
        if self._sync_task:
            self._sync_task.cancel()
        self._save_state()
    
    def register_service_profile(self, profile: ServiceEnergyProfile):
        """Register energy profile for a service"""
        self.service_profiles[profile.service_name] = profile
        
        # Initialize allocation if not present
        if profile.service_name not in self.state.service_allocations:
            # Equal share by default
            num_services = len(self.service_profiles)
            share = self.state.total_budget_kwh / max(num_services, 1)
            self.state.service_allocations[profile.service_name] = share
    
    def update_measurement(self, measurement: EnergyMeasurement):
        """Update energy measurement for a service"""
        with self._lock:
            self.current_measurements[measurement.service_name] = measurement
            
            # Add to history
            self.state.measurements.append(measurement.to_dict())
            
            # Update consumed energy (integrate power over time)
            if measurement.service_name in self.current_measurements:
                # Simple integration: assume constant power since last measurement
                # This is a simplification - real implementation would use better integration
                hours = 1 / 3600  # Assume 1-second intervals
                energy_kwh = (measurement.power_watts / 1000) * hours
                self.state.consumed_kwh += energy_kwh
            
            # Check if sync needed
            if (datetime.now() - self._last_sync).total_seconds() > self.sync_interval:
                self._save_state()
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        now = datetime.now()
        
        # Check if period expired
        if now > self.state.period_end:
            self._start_new_period()
        
        # Calculate projections
        elapsed = (now - self.state.period_start).total_seconds()
        total_seconds = (self.state.period_end - self.state.period_start).total_seconds()
        progress = elapsed / total_seconds
        
        # Current rate
        current_power = sum(m.power_watts for m in self.current_measurements.values())
        
        # Projected consumption
        if progress > 0:
            consumption_rate = self.state.consumed_kwh / progress
            projected_total = consumption_rate
        else:
            projected_total = 0
        
        return {
            "period": {
                "start": self.state.period_start.isoformat(),
                "end": self.state.period_end.isoformat(),
                "progress_percent": progress * 100
            },
            "budget": {
                "total_kwh": self.state.total_budget_kwh,
                "consumed_kwh": self.state.consumed_kwh,
                "remaining_kwh": self.state.total_budget_kwh - self.state.consumed_kwh,
                "consumed_percent": (self.state.consumed_kwh / self.state.total_budget_kwh) * 100
            },
            "current": {
                "power_watts": current_power,
                "services": len(self.current_measurements)
            },
            "projection": {
                "total_kwh": projected_total,
                "over_budget": projected_total > self.state.total_budget_kwh,
                "over_budget_kwh": max(0, projected_total - self.state.total_budget_kwh)
            },
            "allocations": self.state.service_allocations
        }
    
    def _start_new_period(self):
        """Start a new budget period"""
        now = datetime.now()
        
        # Log previous period
        logger.info(f"Period ended: consumed {self.state.consumed_kwh:.2f}/{self.state.total_budget_kwh:.2f} kWh")
        
        # Reset for new period (monthly)
        self.state.period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.state.period_end = (now.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        self.state.consumed_kwh = 0.0
        self.state.measurements = []
        self.state.warnings_sent = []
        
        self._save_state()
    
    def check_warnings(self) -> List[Dict[str, Any]]:
        """Check for budget warnings"""
        warnings = []
        status = self.get_budget_status()
        
        # Check consumption threshold warnings
        thresholds = [50, 75, 90, 95]
        consumed_pct = status["budget"]["consumed_percent"]
        
        for threshold in thresholds:
            if consumed_pct >= threshold:
                warning_key = f"threshold_{threshold}"
                if warning_key not in self.state.warnings_sent:
                    warnings.append({
                        "type": "threshold",
                        "severity": "high" if threshold >= 90 else "medium",
                        "message": f"Energy budget {threshold}% consumed",
                        "threshold": threshold,
                        "consumed_percent": consumed_pct
                    })
                    self.state.warnings_sent.append(warning_key)
        
        # Check projection warnings
        if status["projection"]["over_budget"]:
            warning_key = "projection_over"
            if warning_key not in self.state.warnings_sent:
                warnings.append({
                    "type": "projection",
                    "severity": "high",
                    "message": "Projected to exceed energy budget",
                    "projected_overage_kwh": status["projection"]["over_budget_kwh"]
                })
                self.state.warnings_sent.append(warning_key)
        
        return warnings
    
    def get_service_consumption(self, service_name: str, 
                               hours: int = 24) -> Dict[str, Any]:
        """Get consumption history for a service"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter measurements
        service_measurements = [
            m for m in self.state.measurements
            if m["service_name"] == service_name 
            and datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        if not service_measurements:
            return {
                "service": service_name,
                "hours": hours,
                "measurements": 0,
                "average_power_watts": 0,
                "total_kwh": 0
            }
        
        # Calculate statistics
        total_power = sum(m["power_watts"] for m in service_measurements)
        avg_power = total_power / len(service_measurements)
        
        # Estimate total energy (simple integration)
        total_kwh = (avg_power / 1000) * hours
        
        return {
            "service": service_name,
            "hours": hours,
            "measurements": len(service_measurements),
            "average_power_watts": avg_power,
            "total_kwh": total_kwh,
            "allocation_kwh": self.state.service_allocations.get(service_name, 0),
            "allocation_used_percent": (total_kwh / self.state.service_allocations.get(service_name, 1)) * 100
        }
    
    def set_service_allocation(self, service_name: str, allocation_kwh: float):
        """Set energy allocation for a service"""
        with self._lock:
            self.state.service_allocations[service_name] = allocation_kwh
            self._save_state()
    
    def redistribute_budget(self, weights: Dict[str, float]):
        """Redistribute budget based on weights"""
        total_weight = sum(weights.values())
        
        with self._lock:
            for service, weight in weights.items():
                share = (weight / total_weight) * self.state.total_budget_kwh
                self.state.service_allocations[service] = share
            
            self._save_state()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_energy_budget():
        # Create budget manager
        budget = EnergyBudget()
        
        # Register service profiles
        budget.register_service_profile(ServiceEnergyProfile(
            service_name="tori-ingest",
            baseline_watts=50,
            cpu_watts_per_percent=0.8,
            gpu_watts=150
        ))
        
        budget.register_service_profile(ServiceEnergyProfile(
            service_name="tori-transform",
            baseline_watts=30,
            cpu_watts_per_percent=0.5
        ))
        
        # Start sync task
        await budget.start_sync_task()
        
        # Simulate measurements
        for i in range(5):
            measurement = EnergyMeasurement(
                timestamp=datetime.now(),
                service_name="tori-ingest",
                power_watts=125.5,
                cpu_percent=50,
                memory_gb=2.5,
                io_mbps=100,
                gpu_active=True
            )
            budget.update_measurement(measurement)
            
            # Check status
            status = budget.get_budget_status()
            print(f"Budget status: {status['budget']['consumed_percent']:.1f}% consumed")
            
            # Check warnings
            warnings = budget.check_warnings()
            for warning in warnings:
                print(f"WARNING: {warning['message']}")
            
            await asyncio.sleep(1)
        
        # Get service consumption
        consumption = budget.get_service_consumption("tori-ingest", hours=1)
        print(f"Service consumption: {consumption}")
        
        # Stop sync
        budget.stop_sync_task()
    
    asyncio.run(test_energy_budget())
