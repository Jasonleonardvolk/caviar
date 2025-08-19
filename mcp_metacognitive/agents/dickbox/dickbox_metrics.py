"""
Dickbox Metrics Exporter
========================

Prometheus metrics for Dickbox deployment system.
"""

from fastapi import FastAPI, APIRouter
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Define Prometheus metrics
dickbox_deployments_total = Counter(
    'dickbox_deployments_total',
    'Total number of deployments',
    ['service', 'status']
)

dickbox_rollbacks_total = Counter(
    'dickbox_rollbacks_total',
    'Total number of rollbacks',
    ['service', 'reason']
)

dickbox_active_capsules = Gauge(
    'dickbox_active_capsules',
    'Number of active capsules'
)

dickbox_running_services = Gauge(
    'dickbox_running_services',
    'Number of running services'
)

dickbox_capsule_size_bytes = Gauge(
    'dickbox_capsule_size_bytes',
    'Size of deployed capsules in bytes',
    ['capsule_id']
)

dickbox_deployment_duration_seconds = Histogram(
    'dickbox_deployment_duration_seconds',
    'Time taken for deployments',
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
)

dickbox_health_check_duration_seconds = Histogram(
    'dickbox_health_check_duration_seconds',
    'Time taken for health checks',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

# GPU metrics
dickbox_gpu_count = Gauge(
    'dickbox_gpu_count',
    'Number of available GPUs'
)

dickbox_gpu_memory_used_mb = Gauge(
    'dickbox_gpu_memory_used_mb',
    'GPU memory used in MB',
    ['gpu_index', 'gpu_name']
)

dickbox_gpu_utilization_percent = Gauge(
    'dickbox_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_index', 'gpu_name']
)

dickbox_gpu_temperature_celsius = Gauge(
    'dickbox_gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['gpu_index', 'gpu_name']
)

dickbox_mps_active = Gauge(
    'dickbox_mps_active',
    'Whether NVIDIA MPS is active'
)

# Slice metrics
dickbox_slice_cpu_usage_seconds = Gauge(
    'dickbox_slice_cpu_usage_seconds',
    'CPU usage by systemd slice',
    ['slice']
)

dickbox_slice_memory_bytes = Gauge(
    'dickbox_slice_memory_bytes',
    'Memory usage by systemd slice',
    ['slice']
)

dickbox_slice_task_count = Gauge(
    'dickbox_slice_task_count',
    'Number of tasks in systemd slice',
    ['slice']
)

# Communication metrics
dickbox_ipc_requests_total = Counter(
    'dickbox_ipc_requests_total',
    'Total IPC requests',
    ['service', 'method']
)

dickbox_ipc_request_duration_seconds = Histogram(
    'dickbox_ipc_request_duration_seconds',
    'IPC request duration',
    ['service', 'method'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
)

dickbox_pubsub_messages_total = Counter(
    'dickbox_pubsub_messages_total',
    'Total pub/sub messages',
    ['topic', 'direction']  # direction: published/received
)

# Create metrics router
metrics_router = APIRouter(prefix="/dickbox", tags=["metrics"])


class DickboxMetricsExporter:
    """Bridges Dickbox agent with Prometheus metrics"""
    
    def __init__(self, dickbox_agent=None):
        self.dickbox_agent = dickbox_agent
        self._update_task = None
        self._running = False
        
    def set_dickbox_agent(self, agent):
        """Set or update the Dickbox agent reference"""
        self.dickbox_agent = agent
        
    async def start_metrics_collection(self, update_interval: int = 30):
        """Start periodic metrics collection"""
        if self._running:
            logger.warning("Metrics collection already running")
            return
            
        self._running = True
        self._update_task = asyncio.create_task(self._update_metrics_loop(update_interval))
        logger.info(f"Started Dickbox metrics collection (interval: {update_interval}s)")
        
    async def stop_metrics_collection(self):
        """Stop metrics collection"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped Dickbox metrics collection")
        
    async def _update_metrics_loop(self, interval: int):
        """Periodically update Prometheus metrics from Dickbox state"""
        while self._running:
            try:
                await self._update_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating Dickbox metrics: {e}")
                await asyncio.sleep(interval)
                
    async def _update_metrics(self):
        """Update all Prometheus metrics from current Dickbox state"""
        if not self.dickbox_agent:
            logger.warning("No Dickbox agent set for metrics export")
            return
            
        try:
            # Get system status
            status = await self.dickbox_agent._get_status()
            
            # Update capsule metrics
            dickbox_active_capsules.set(status["capsules"]["active"])
            dickbox_running_services.set(status["services"]["running"])
            
            # Update GPU metrics if available
            if hasattr(self.dickbox_agent, 'gpu_manager'):
                gpu_info = await self.dickbox_agent.gpu_manager.get_gpu_info()
                dickbox_gpu_count.set(len(gpu_info))
                
                for gpu in gpu_info:
                    labels = {'gpu_index': str(gpu.index), 'gpu_name': gpu.name}
                    dickbox_gpu_memory_used_mb.labels(**labels).set(gpu.memory_used)
                    dickbox_gpu_utilization_percent.labels(**labels).set(gpu.utilization)
                    dickbox_gpu_temperature_celsius.labels(**labels).set(gpu.temperature)
                
                # Check MPS status
                mps_running = await self.dickbox_agent.gpu_manager.is_mps_running()
                dickbox_mps_active.set(1 if mps_running else 0)
            
            # Update slice metrics if available
            if hasattr(self.dickbox_agent, 'slice_manager'):
                for slice_name in ['tori.slice', 'tori-server.slice', 'tori-helper.slice', 'tori-build.slice']:
                    try:
                        slice_status = await self.dickbox_agent.slice_manager.get_slice_status(slice_name)
                        if slice_status.get('active'):
                            dickbox_slice_cpu_usage_seconds.labels(slice=slice_name).set(
                                slice_status.get('cpu_usage_ns', 0) / 1e9
                            )
                            dickbox_slice_memory_bytes.labels(slice=slice_name).set(
                                slice_status.get('memory_bytes', 0)
                            )
                            dickbox_slice_task_count.labels(slice=slice_name).set(
                                slice_status.get('task_count', 0)
                            )
                    except Exception as e:
                        logger.debug(f"Failed to get metrics for slice {slice_name}: {e}")
            
            logger.debug("Updated Dickbox metrics")
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def record_deployment(self, service: str, status: str, duration: float = None):
        """Record a deployment event"""
        dickbox_deployments_total.labels(service=service, status=status).inc()
        if duration is not None:
            dickbox_deployment_duration_seconds.observe(duration)
    
    def record_rollback(self, service: str, reason: str):
        """Record a rollback event"""
        dickbox_rollbacks_total.labels(service=service, reason=reason).inc()
    
    def record_health_check(self, duration: float):
        """Record health check duration"""
        dickbox_health_check_duration_seconds.observe(duration)
    
    def record_ipc_request(self, service: str, method: str, duration: float):
        """Record IPC request metrics"""
        dickbox_ipc_requests_total.labels(service=service, method=method).inc()
        dickbox_ipc_request_duration_seconds.labels(service=service, method=method).observe(duration)
    
    def record_pubsub_message(self, topic: str, direction: str):
        """Record pub/sub message"""
        dickbox_pubsub_messages_total.labels(topic=topic, direction=direction).inc()


# Global exporter instance
metrics_exporter = DickboxMetricsExporter()


@metrics_router.get("/metrics", response_class=Response)
async def get_metrics():
    """Prometheus metrics endpoint"""
    # Force update before serving metrics
    await metrics_exporter._update_metrics()
    
    # Generate Prometheus format metrics
    metrics_data = generate_latest()
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


@metrics_router.get("/health")
async def health_check():
    """Health check endpoint"""
    if not metrics_exporter.dickbox_agent:
        return {"status": "unhealthy", "reason": "No Dickbox agent connected"}
    
    try:
        status = await metrics_exporter.dickbox_agent._get_status()
        return {
            "status": "healthy",
            "capsules": status["capsules"],
            "services": status["services"],
            "gpu": status.get("gpu", {})
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "reason": str(e)
        }


@metrics_router.get("/slices")
async def get_slice_status():
    """Get systemd slice status"""
    if not metrics_exporter.dickbox_agent:
        return {"error": "No Dickbox agent connected"}
    
    if not hasattr(metrics_exporter.dickbox_agent, 'slice_manager'):
        return {"error": "Slice manager not available"}
    
    slices = {}
    for slice_name in ['tori.slice', 'tori-server.slice', 'tori-helper.slice', 'tori-build.slice']:
        try:
            status = await metrics_exporter.dickbox_agent.slice_manager.get_slice_status(slice_name)
            slices[slice_name] = status
        except Exception as e:
            slices[slice_name] = {"error": str(e)}
    
    return {"slices": slices}


@metrics_router.get("/gpu")
async def get_gpu_status():
    """Get GPU status and allocation"""
    if not metrics_exporter.dickbox_agent:
        return {"error": "No Dickbox agent connected"}
    
    if not hasattr(metrics_exporter.dickbox_agent, 'gpu_manager'):
        return {"error": "GPU manager not available"}
    
    try:
        gpus = await metrics_exporter.dickbox_agent.gpu_manager.get_gpu_info()
        mps_running = await metrics_exporter.dickbox_agent.gpu_manager.is_mps_running()
        
        return {
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory": {
                        "total_mb": gpu.memory_total,
                        "used_mb": gpu.memory_used,
                        "free_mb": gpu.memory_free
                    },
                    "utilization": gpu.utilization,
                    "temperature": gpu.temperature,
                    "processes": gpu.processes
                }
                for gpu in gpus
            ],
            "mps_active": mps_running
        }
    except Exception as e:
        return {"error": str(e)}


def create_metrics_app(dickbox_agent=None) -> FastAPI:
    """Create FastAPI app with metrics endpoints"""
    app = FastAPI(title="Dickbox Metrics", version="1.0.0")
    
    # Include metrics router
    app.include_router(metrics_router)
    
    # Set Dickbox agent if provided
    if dickbox_agent:
        metrics_exporter.set_dickbox_agent(dickbox_agent)
    
    @app.on_event("startup")
    async def startup_event():
        """Start metrics collection on app startup"""
        if dickbox_agent:
            await metrics_exporter.start_metrics_collection()
            logger.info("Dickbox metrics exporter started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Stop metrics collection on shutdown"""
        await metrics_exporter.stop_metrics_collection()
        logger.info("Dickbox metrics exporter stopped")
    
    return app


# Export
__all__ = [
    'metrics_exporter',
    'create_metrics_app',
    'metrics_router',
    'DickboxMetricsExporter'
]
