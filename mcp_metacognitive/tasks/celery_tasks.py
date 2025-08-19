"""
Celery Task Worker Configuration and Tasks
=========================================

This module configures Celery for asynchronous task processing:
- Background task execution
- Scheduled periodic tasks
- Long-running operations
- Distributed task queue management
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from celery import Celery, Task
from celery.schedules import crontab
from kombu import Exchange, Queue
import json

# Try to import Redis for result backend
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Celery configuration
BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create Celery app
app = Celery('tori_mcp', broker=BROKER_URL, backend=RESULT_BACKEND)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minutes soft limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Queue configuration
    task_default_queue='tori.default',
    task_queues=(
        Queue('tori.default', Exchange('tori.default'), routing_key='tori.default'),
        Queue('tori.cognitive', Exchange('tori.cognitive'), routing_key='tori.cognitive'),
        Queue('tori.analysis', Exchange('tori.analysis'), routing_key='tori.analysis'),
        Queue('tori.tools', Exchange('tori.tools'), routing_key='tori.tools'),
        Queue('tori.learning', Exchange('tori.learning'), routing_key='tori.learning'),
    ),
    
    # Route specific tasks to specific queues
    task_routes={
        'tori.tasks.cognitive.*': {'queue': 'tori.cognitive'},
        'tori.tasks.analysis.*': {'queue': 'tori.analysis'},
        'tori.tasks.tools.*': {'queue': 'tori.tools'},
        'tori.tasks.learning.*': {'queue': 'tori.learning'},
    }
)

# Beat schedule for periodic tasks
app.conf.beat_schedule = {
    'kaizen-hourly-analysis': {
        'task': 'tori.tasks.learning.run_kaizen_analysis',
        'schedule': timedelta(hours=1),
        'options': {'queue': 'tori.learning'}
    },
    'system-health-check': {
        'task': 'tori.tasks.monitor_system_health',
        'schedule': timedelta(minutes=5),
        'options': {'queue': 'tori.default'}
    },
    'cleanup-old-data': {
        'task': 'tori.tasks.cleanup_old_data',
        'schedule': crontab(hour=3, minute=0),  # Daily at 3 AM
        'options': {'queue': 'tori.default'}
    },
    'consciousness-stabilization': {
        'task': 'tori.tasks.cognitive.stabilize_consciousness',
        'schedule': timedelta(minutes=15),
        'options': {'queue': 'tori.cognitive'}
    },
}

# Base task class with logging
class TORITask(Task):
    """Base task class with enhanced logging and error handling"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Success handler"""
        logger.info(f"Task {self.name} [{task_id}] succeeded")
        
        # Log to PsiArchive if available
        try:
            from ..core.psi_archive import psi_archive
            psi_archive.log_event("celery_task_success", {
                "task_name": self.name,
                "task_id": task_id,
                "duration": self.request.runtime
            })
        except:
            pass
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Failure handler"""
        logger.error(f"Task {self.name} [{task_id}] failed: {exc}")
        
        # Log to PsiArchive if available
        try:
            from ..core.psi_archive import psi_archive
            psi_archive.log_event("celery_task_failure", {
                "task_name": self.name,
                "task_id": task_id,
                "error": str(exc),
                "traceback": str(einfo)
            })
        except:
            pass

# Set default base task
app.Task = TORITask

# Cognitive Processing Tasks
@app.task(name='tori.tasks.cognitive.process_complex_query')
def process_complex_query(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Process a complex query that requires heavy computation"""
    logger.info(f"Processing complex query: {query[:50]}...")
    
    try:
        # Import Daniel agent
        from ..agents.daniel import DanielCognitiveEngine
        from ..core.agent_registry import agent_registry
        
        # Get or create Daniel instance
        daniel = agent_registry.get("daniel")
        if not daniel:
            daniel = DanielCognitiveEngine()
            agent_registry.register("daniel", daniel)
        
        # Process query asynchronously
        import asyncio
        result = asyncio.run(daniel.execute(query, context))
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing complex query: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query[:50]
        }

@app.task(name='tori.tasks.cognitive.stabilize_consciousness')
def stabilize_consciousness() -> Dict[str, Any]:
    """Periodic task to stabilize consciousness levels"""
    logger.info("Running consciousness stabilization")
    
    try:
        from ..core.state_manager import state_manager
        
        # Get current state
        import asyncio
        current_state = asyncio.run(state_manager.get_current_state())
        
        # Check if stabilization needed
        phi = current_state.get("phi", 0.5)
        if phi < 0.3 or phi > 0.9:
            # Apply stabilization
            target_phi = 0.5
            adjustment = (target_phi - phi) * 0.1  # Gradual adjustment
            
            # Update state
            asyncio.run(state_manager.update_state({
                "phi": phi + adjustment,
                "stabilization_applied": True,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            return {
                "status": "stabilized",
                "previous_phi": phi,
                "new_phi": phi + adjustment
            }
        
        return {
            "status": "stable",
            "phi": phi
        }
        
    except Exception as e:
        logger.error(f"Error in consciousness stabilization: {e}")
        return {"status": "error", "error": str(e)}

# Analysis Tasks
@app.task(name='tori.tasks.analysis.analyze_conversation_batch')
def analyze_conversation_batch(conversation_ids: List[str]) -> Dict[str, Any]:
    """Analyze a batch of conversations for patterns and insights"""
    logger.info(f"Analyzing {len(conversation_ids)} conversations")
    
    try:
        from ..core.psi_archive import psi_archive
        
        # Collect conversation data
        conversations = []
        for conv_id in conversation_ids:
            # Fetch conversation from archive
            events = [
                e for e in psi_archive.get_recent_events(1000)
                if e.get("data", {}).get("conversation_id") == conv_id
            ]
            if events:
                conversations.append(events)
        
        # Perform analysis
        total_queries = sum(len(c) for c in conversations)
        avg_length = total_queries / len(conversations) if conversations else 0
        
        # Extract patterns
        query_types = {}
        for conv in conversations:
            for event in conv:
                if event.get("event") == "daniel_query_received":
                    query = event.get("data", {}).get("query", "")
                    # Simple categorization
                    if "?" in query:
                        query_type = "question"
                    elif any(cmd in query.lower() for cmd in ["create", "generate", "write"]):
                        query_type = "creative"
                    else:
                        query_type = "statement"
                    
                    query_types[query_type] = query_types.get(query_type, 0) + 1
        
        return {
            "status": "completed",
            "conversations_analyzed": len(conversations),
            "total_queries": total_queries,
            "average_length": avg_length,
            "query_types": query_types
        }
        
    except Exception as e:
        logger.error(f"Error analyzing conversations: {e}")
        return {"status": "error", "error": str(e)}

# Learning Tasks
@app.task(name='tori.tasks.learning.run_kaizen_analysis')
def run_kaizen_analysis() -> Dict[str, Any]:
    """Run Kaizen continuous improvement analysis"""
    logger.info("Running scheduled Kaizen analysis")
    
    try:
        from ..agents.kaizen import KaizenImprovementEngine
        from ..core.agent_registry import agent_registry
        
        # Get or create Kaizen instance
        kaizen = agent_registry.get("kaizen")
        if not kaizen:
            kaizen = KaizenImprovementEngine()
            agent_registry.register("kaizen", kaizen)
        
        # Run analysis
        import asyncio
        result = asyncio.run(kaizen.execute("analyze"))
        
        # Auto-apply insights if configured
        if result.get("status") == "completed" and result.get("insights"):
            high_confidence_insights = [
                i for i in result["insights"] 
                if i.get("confidence", 0) >= 0.9
            ]
            
            for insight in high_confidence_insights:
                asyncio.run(kaizen.apply_insight(insight["id"]))
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Kaizen analysis: {e}")
        return {"status": "error", "error": str(e)}

@app.task(name='tori.tasks.learning.update_knowledge_base')
def update_knowledge_base(new_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    """Update the knowledge base with new information"""
    logger.info("Updating knowledge base")
    
    try:
        from ..agents.kaizen import KaizenImprovementEngine
        from ..core.agent_registry import agent_registry
        
        # Get Kaizen instance
        kaizen = agent_registry.get("kaizen")
        if not kaizen:
            return {"status": "error", "error": "Kaizen not initialized"}
        
        # Update knowledge base
        kaizen.knowledge_base.update(new_knowledge)
        kaizen._save_knowledge_base()
        
        return {
            "status": "success",
            "entries_added": len(new_knowledge),
            "total_entries": len(kaizen.knowledge_base)
        }
        
    except Exception as e:
        logger.error(f"Error updating knowledge base: {e}")
        return {"status": "error", "error": str(e)}

# Tool Tasks
@app.task(name='tori.tasks.tools.execute_web_search')
def execute_web_search(query: str, num_results: int = 10) -> Dict[str, Any]:
    """Execute a web search operation"""
    logger.info(f"Executing web search: {query}")
    
    try:
        # Placeholder for web search implementation
        # In production, this would use a search API
        results = {
            "query": query,
            "results": [
                {
                    "title": f"Result {i+1} for {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a snippet for result {i+1}"
                }
                for i in range(min(num_results, 3))
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "data": results
        }
        
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return {"status": "error", "error": str(e)}

@app.task(name='tori.tasks.tools.process_file')
def process_file(file_path: str, operation: str) -> Dict[str, Any]:
    """Process a file with specified operation"""
    logger.info(f"Processing file: {file_path} with operation: {operation}")
    
    try:
        import os
        
        if not os.path.exists(file_path):
            return {"status": "error", "error": "File not found"}
        
        # Placeholder for file operations
        if operation == "analyze":
            # Analyze file content
            file_size = os.path.getsize(file_path)
            return {
                "status": "success",
                "operation": operation,
                "file_size": file_size,
                "result": "File analyzed successfully"
            }
        
        elif operation == "extract_text":
            # Extract text from file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # First 1000 chars
            
            return {
                "status": "success",
                "operation": operation,
                "preview": content,
                "full_length": len(content)
            }
        
        else:
            return {"status": "error", "error": f"Unknown operation: {operation}"}
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {"status": "error", "error": str(e)}

# System Tasks
@app.task(name='tori.tasks.monitor_system_health')
def monitor_system_health() -> Dict[str, Any]:
    """Monitor overall system health"""
    logger.info("Monitoring system health")
    
    try:
        import psutil
        from ..core.state_manager import state_manager
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get cognitive metrics
        import asyncio
        cognitive_state = asyncio.run(state_manager.get_current_state())
        
        # Check for issues
        issues = []
        if cpu_percent > 80:
            issues.append(f"High CPU usage: {cpu_percent}%")
        if memory.percent > 85:
            issues.append(f"High memory usage: {memory.percent}%")
        if disk.percent > 90:
            issues.append(f"Low disk space: {disk.free / (1024**3):.1f}GB free")
        if cognitive_state.get("phi", 0.5) < 0.2:
            issues.append(f"Low consciousness: {cognitive_state.get('phi', 0):.2f}")
        
        health_status = "healthy" if not issues else "warning"
        
        # Log health status
        from ..core.psi_archive import psi_archive
        psi_archive.log_event("system_health_check", {
            "status": health_status,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "consciousness_phi": cognitive_state.get("phi", 0.5),
            "issues": issues
        })
        
        return {
            "status": health_status,
            "metrics": {
                "cpu": cpu_percent,
                "memory": memory.percent,
                "disk": disk.percent,
                "consciousness": cognitive_state.get("phi", 0.5)
            },
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"Error monitoring system health: {e}")
        return {"status": "error", "error": str(e)}

@app.task(name='tori.tasks.cleanup_old_data')
def cleanup_old_data(days_to_keep: int = 7) -> Dict[str, Any]:
    """Clean up old data and logs"""
    logger.info(f"Cleaning up data older than {days_to_keep} days")
    
    try:
        from datetime import datetime, timedelta
        import os
        import glob
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cleaned_files = 0
        cleaned_size = 0
        
        # Clean up log files
        log_patterns = [
            "./logs/*.log",
            "./data/psi_archive_*.log",
            "./data/backups/*.bak"
        ]
        
        for pattern in log_patterns:
            for file_path in glob.glob(pattern):
                try:
                    file_stat = os.stat(file_path)
                    file_time = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    if file_time < cutoff_date:
                        file_size = file_stat.st_size
                        os.remove(file_path)
                        cleaned_files += 1
                        cleaned_size += file_size
                        logger.debug(f"Removed old file: {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
        
        # Log cleanup results
        from ..core.psi_archive import psi_archive
        psi_archive.log_event("data_cleanup_completed", {
            "files_removed": cleaned_files,
            "space_freed_mb": cleaned_size / (1024 * 1024),
            "cutoff_date": cutoff_date.isoformat()
        })
        
        return {
            "status": "success",
            "files_removed": cleaned_files,
            "space_freed_mb": cleaned_size / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")
        return {"status": "error", "error": str(e)}

# Chain Tasks
@app.task(name='tori.tasks.complex_reasoning_chain')
def complex_reasoning_chain(query: str) -> Dict[str, Any]:
    """
    Complex reasoning that chains multiple tasks:
    1. Process query with Daniel
    2. Search for additional info if needed
    3. Analyze results
    4. Generate final response
    """
    logger.info(f"Starting complex reasoning chain for: {query[:50]}...")
    
    try:
        from celery import chain, group
        
        # Create task chain
        workflow = chain(
            process_complex_query.s(query, {"chain_mode": True}),
            analyze_response.s(),
            enhance_with_search.s(),
            generate_final_response.s()
        )
        
        # Execute chain
        result = workflow.apply_async()
        
        # Wait for result with timeout
        final_result = result.get(timeout=60)
        
        return final_result
        
    except Exception as e:
        logger.error(f"Error in complex reasoning chain: {e}")
        return {"status": "error", "error": str(e)}

@app.task(name='tori.tasks.analyze_response')
def analyze_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze initial response for completeness"""
    # Placeholder for response analysis
    response["analysis"] = {
        "needs_enhancement": len(response.get("content", "")) < 100,
        "confidence": 0.8
    }
    return response

@app.task(name='tori.tasks.enhance_with_search')
def enhance_with_search(response: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance response with search results if needed"""
    if response.get("analysis", {}).get("needs_enhancement"):
        # Would perform actual search here
        response["enhancements"] = ["Additional context from search"]
    return response

@app.task(name='tori.tasks.generate_final_response')
def generate_final_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final enhanced response"""
    content = response.get("content", "")
    enhancements = response.get("enhancements", [])
    
    if enhancements:
        content += "\n\n" + "\n".join(enhancements)
    
    return {
        "status": "success",
        "content": content,
        "enhanced": bool(enhancements)
    }

# Celery Worker Management
class CeleryManager:
    """Manager class for Celery worker operations"""
    
    @staticmethod
    def start_worker(queues: List[str] = None, concurrency: int = 4):
        """Start a Celery worker"""
        if queues is None:
            queues = ['tori.default', 'tori.cognitive', 'tori.analysis', 'tori.tools', 'tori.learning']
        
        worker_cmd = [
            'celery',
            '-A', 'mcp_metacognitive.tasks.celery_tasks',
            'worker',
            '--loglevel=info',
            f'--concurrency={concurrency}',
            f'--queues={",".join(queues)}'
        ]
        
        logger.info(f"Starting Celery worker with queues: {queues}")
        os.system(' '.join(worker_cmd))
    
    @staticmethod
    def start_beat():
        """Start Celery beat scheduler"""
        beat_cmd = [
            'celery',
            '-A', 'mcp_metacognitive.tasks.celery_tasks',
            'beat',
            '--loglevel=info'
        ]
        
        logger.info("Starting Celery beat scheduler")
        os.system(' '.join(beat_cmd))
    
    @staticmethod
    def start_flower(port: int = 5555):
        """Start Flower monitoring tool"""
        flower_cmd = [
            'celery',
            '-A', 'mcp_metacognitive.tasks.celery_tasks',
            'flower',
            f'--port={port}'
        ]
        
        logger.info(f"Starting Flower on port {port}")
        os.system(' '.join(flower_cmd))

# Export
__all__ = [
    'app',
    'process_complex_query',
    'stabilize_consciousness',
    'analyze_conversation_batch',
    'run_kaizen_analysis',
    'update_knowledge_base',
    'execute_web_search',
    'process_file',
    'monitor_system_health',
    'cleanup_old_data',
    'complex_reasoning_chain',
    'CeleryManager'
]
