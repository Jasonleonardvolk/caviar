"""
MCP Metacognitive Server - Production Implementation
Provides metacognitive capabilities for the TORI/KHA system
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetacognitiveState:
    """Current metacognitive state"""
    awareness_level: float  # 0-1
    confidence: float  # 0-1
    uncertainty: float  # 0-1
    cognitive_load: float  # 0-1
    reflection_depth: int  # 0-N
    active_strategies: List[str]
    performance_history: List[float]
    timestamp: datetime

@dataclass
class CognitiveStrategy:
    """A cognitive strategy that can be applied"""
    name: str
    description: str
    applicability: Callable[[Any], float]  # Returns 0-1
    execute: Callable[[Any], Any]
    cost: float  # Cognitive cost
    success_rate: float  # Historical success

class MCPMetacognitiveServer:
    """
    Metacognitive server for monitoring and optimizing cognitive processes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Server configuration
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 8888)
        
        # Metacognitive state
        self.state = MetacognitiveState(
            awareness_level=0.5,
            confidence=0.5,
            uncertainty=0.5,
            cognitive_load=0.0,
            reflection_depth=0,
            active_strategies=[],
            performance_history=[],
            timestamp=datetime.now()
        )
        
        # Available strategies
        self.strategies: Dict[str, CognitiveStrategy] = {}
        self._register_default_strategies()
        
        # Performance tracking
        self.task_history: List[Dict[str, Any]] = []
        self.performance_window = self.config.get('performance_window', 100)
        
        # Thresholds
        self.load_threshold = self.config.get('load_threshold', 0.8)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.7)
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info(f"MCP Metacognitive Server initialized on {self.host}:{self.port}")
    
    def _register_default_strategies(self):
        """Register default cognitive strategies"""
        
        # Strategy: Decomposition
        self.register_strategy(CognitiveStrategy(
            name="decomposition",
            description="Break complex problems into smaller parts",
            applicability=lambda task: task.get('complexity', 0) > 0.7,
            execute=self._decompose_task,
            cost=0.3,
            success_rate=0.85
        ))
        
        # Strategy: Reflection
        self.register_strategy(CognitiveStrategy(
            name="reflection",
            description="Reflect on recent decisions and outcomes",
            applicability=lambda task: self.state.uncertainty > 0.6,
            execute=self._reflect_on_performance,
            cost=0.2,
            success_rate=0.9
        ))
        
        # Strategy: Simplification
        self.register_strategy(CognitiveStrategy(
            name="simplification",
            description="Simplify the problem representation",
            applicability=lambda task: self.state.cognitive_load > 0.7,
            execute=self._simplify_task,
            cost=0.1,
            success_rate=0.8
        ))
        
        # Strategy: Pattern Recognition
        self.register_strategy(CognitiveStrategy(
            name="pattern_recognition",
            description="Look for patterns in similar past tasks",
            applicability=lambda task: len(self.task_history) > 10,
            execute=self._find_patterns,
            cost=0.4,
            success_rate=0.75
        ))
        
        # Strategy: Resource Reallocation
        self.register_strategy(CognitiveStrategy(
            name="resource_reallocation",
            description="Reallocate cognitive resources",
            applicability=lambda task: self.state.cognitive_load > 0.9,
            execute=self._reallocate_resources,
            cost=0.5,
            success_rate=0.7
        ))
    
    def register_strategy(self, strategy: CognitiveStrategy):
        """Register a new cognitive strategy"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name}")
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task with metacognitive oversight"""
        
        # Update cognitive load based on task
        task_complexity = task.get('complexity', 0.5)
        self.state.cognitive_load = min(1.0, self.state.cognitive_load * 0.9 + task_complexity * 0.3)
        
        # Select applicable strategies
        selected_strategies = self._select_strategies(task)
        
        # Apply strategies
        result = task.copy()
        for strategy_name in selected_strategies:
            strategy = self.strategies[strategy_name]
            try:
                result = strategy.execute(result)
                self.state.active_strategies.append(strategy_name)
                logger.info(f"Applied strategy: {strategy_name}")
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {e}")
        
        # Update metacognitive state
        self._update_state_after_task(task, result)
        
        # Record task for history
        self.task_history.append({
            'task': task,
            'result': result,
            'strategies': selected_strategies,
            'state': asdict(self.state),
            'timestamp': datetime.now()
        })
        
        # Trim history
        if len(self.task_history) > self.performance_window:
            self.task_history = self.task_history[-self.performance_window:]
        
        return {
            'result': result,
            'metacognitive': {
                'strategies_used': selected_strategies,
                'confidence': self.state.confidence,
                'cognitive_load': self.state.cognitive_load,
                'awareness_level': self.state.awareness_level
            }
        }
    
    def _select_strategies(self, task: Dict[str, Any]) -> List[str]:
        """Select appropriate strategies for the task"""
        selected = []
        
        # Evaluate each strategy
        strategy_scores = []
        for name, strategy in self.strategies.items():
            applicability = strategy.applicability(task)
            if applicability > 0.5:  # Threshold for consideration
                # Score based on applicability, success rate, and cognitive cost
                score = (applicability * strategy.success_rate) / (1 + strategy.cost)
                strategy_scores.append((name, score))
        
        # Sort by score and select top strategies
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select strategies that fit within cognitive budget
        remaining_capacity = 1.0 - self.state.cognitive_load
        for name, score in strategy_scores:
            strategy = self.strategies[name]
            if strategy.cost <= remaining_capacity:
                selected.append(name)
                remaining_capacity -= strategy.cost
        
        return selected
    
    def _update_state_after_task(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Update metacognitive state after task completion"""
        
        # Calculate task success (simplified)
        success = result.get('success', 0.5)
        self.state.performance_history.append(success)
        
        # Update confidence based on recent performance
        if len(self.state.performance_history) > 5:
            recent_performance = np.mean(self.state.performance_history[-5:])
            self.state.confidence = 0.8 * self.state.confidence + 0.2 * recent_performance
        
        # Update uncertainty
        if len(self.state.performance_history) > 10:
            performance_std = np.std(self.state.performance_history[-10:])
            self.state.uncertainty = min(1.0, performance_std * 2)
        
        # Update awareness level
        self.state.awareness_level = min(1.0, 
            0.5 * (1 - self.state.cognitive_load) + 
            0.3 * self.state.confidence + 
            0.2 * (1 - self.state.uncertainty)
        )
        
        # Clear active strategies
        self.state.active_strategies = []
        
        # Update timestamp
        self.state.timestamp = datetime.now()
    
    # Strategy implementations
    def _decompose_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose complex task into subtasks"""
        if 'subtasks' not in task:
            # Simple decomposition based on task description
            task['subtasks'] = []
            if 'description' in task:
                # Split by common delimiters
                parts = task['description'].split(' and ')
                for i, part in enumerate(parts):
                    task['subtasks'].append({
                        'id': f"subtask_{i}",
                        'description': part.strip(),
                        'complexity': task.get('complexity', 0.5) / len(parts)
                    })
        
        task['decomposed'] = True
        return task
    
    def _reflect_on_performance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on recent performance"""
        if len(self.task_history) > 0:
            # Analyze recent tasks
            recent_tasks = self.task_history[-5:]
            patterns = {
                'avg_success': np.mean([t['result'].get('success', 0.5) for t in recent_tasks]),
                'common_strategies': [],
                'task_types': []
            }
            
            # Find common strategies
            strategy_counts = {}
            for t in recent_tasks:
                for strategy in t.get('strategies', []):
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            patterns['common_strategies'] = sorted(
                strategy_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            task['reflection'] = patterns
        
        self.state.reflection_depth += 1
        return task
    
    def _simplify_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify task representation"""
        # Remove non-essential fields
        essential_fields = ['id', 'type', 'description', 'priority', 'complexity']
        simplified = {k: v for k, v in task.items() if k in essential_fields}
        
        # Add simplification flag
        simplified['simplified'] = True
        simplified['original_fields'] = list(task.keys())
        
        return simplified
    
    def _find_patterns(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Find patterns in similar past tasks"""
        similar_tasks = []
        
        task_type = task.get('type', 'unknown')
        for historical_task in self.task_history:
            if historical_task['task'].get('type') == task_type:
                similar_tasks.append(historical_task)
        
        if similar_tasks:
            # Analyze successful strategies
            successful_strategies = {}
            for t in similar_tasks:
                if t['result'].get('success', 0) > 0.7:
                    for strategy in t.get('strategies', []):
                        successful_strategies[strategy] = successful_strategies.get(strategy, 0) + 1
            
            task['pattern_analysis'] = {
                'similar_tasks_found': len(similar_tasks),
                'recommended_strategies': sorted(
                    successful_strategies.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            }
        
        return task
    
    def _reallocate_resources(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Reallocate cognitive resources"""
        # Identify low-priority active processes
        task['resource_reallocation'] = {
            'previous_load': self.state.cognitive_load,
            'freed_resources': 0
        }
        
        # Simulate freeing up resources
        freed = min(0.3, self.state.cognitive_load * 0.4)
        self.state.cognitive_load -= freed
        task['resource_reallocation']['freed_resources'] = freed
        
        return task
    
    def get_state(self) -> Dict[str, Any]:
        """Get current metacognitive state"""
        return asdict(self.state)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.state.performance_history:
            return {'status': 'no_data'}
        
        return {
            'avg_performance': np.mean(self.state.performance_history),
            'std_performance': np.std(self.state.performance_history),
            'trend': self._calculate_trend(),
            'total_tasks': len(self.task_history),
            'strategy_usage': self._calculate_strategy_usage(),
            'current_state': self.get_state()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.state.performance_history) < 10:
            return 'insufficient_data'
        
        recent = np.mean(self.state.performance_history[-5:])
        older = np.mean(self.state.performance_history[-10:-5])
        
        if recent > older + 0.1:
            return 'improving'
        elif recent < older - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_strategy_usage(self) -> Dict[str, int]:
        """Calculate strategy usage statistics"""
        usage = {}
        for task in self.task_history:
            for strategy in task.get('strategies', []):
                usage[strategy] = usage.get(strategy, 0) + 1
        return usage
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Started metacognitive monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped metacognitive monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for concerning states
                if self.state.cognitive_load > self.load_threshold:
                    logger.warning(f"High cognitive load: {self.state.cognitive_load}")
                
                if self.state.confidence < self.confidence_threshold:
                    logger.warning(f"Low confidence: {self.state.confidence}")
                
                if self.state.uncertainty > self.uncertainty_threshold:
                    logger.warning(f"High uncertainty: {self.state.uncertainty}")
                
                # Natural decay of cognitive load
                self.state.cognitive_load *= 0.99
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def serve(self):
        """Start the MCP server"""
        # This would implement the full MCP protocol
        # For now, we'll create a simple HTTP API
        from aiohttp import web
        
        app = web.Application()
        
        # Routes
        app.router.add_post('/process', self._handle_process)
        app.router.add_get('/state', self._handle_state)
        app.router.add_get('/report', self._handle_report)
        app.router.add_post('/strategy', self._handle_strategy)
        
        # Start monitoring
        self.start_monitoring()
        
        # Run server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"MCP Metacognitive Server running on {self.host}:{self.port}")
        
        # Keep running
        try:
            await asyncio.Event().wait()
        finally:
            self.stop_monitoring()
            await runner.cleanup()
    
    async def _handle_process(self, request):
        """Handle task processing request"""
        from aiohttp import web
        
        try:
            data = await request.json()
            result = await self.process_task(data)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_state(self, request):
        """Handle state query"""
        from aiohttp import web
        return web.json_response(self.get_state())
    
    async def _handle_report(self, request):
        """Handle performance report request"""
        from aiohttp import web
        return web.json_response(self.get_performance_report())
    
    async def _handle_strategy(self, request):
        """Handle strategy registration"""
        from aiohttp import web
        
        try:
            data = await request.json()
            # This would need proper deserialization of the strategy
            # For now, just acknowledge
            return web.json_response({'status': 'acknowledged'})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

# Example usage
if __name__ == "__main__":
    async def test_server():
        server = MCPMetacognitiveServer({
            'host': 'localhost',
            'port': 8888
        })
        
        # Test task processing
        test_task = {
            'id': 'test_001',
            'type': 'analysis',
            'description': 'Analyze code structure and identify optimization opportunities',
            'complexity': 0.8,
            'priority': 'high'
        }
        
        result = await server.process_task(test_task)
        print(f"Task result: {json.dumps(result, indent=2, default=str)}")
        
        # Get performance report
        report = server.get_performance_report()
        print(f"Performance report: {json.dumps(report, indent=2)}")
        
        # Run server
        await server.serve()
    
    asyncio.run(test_server())
