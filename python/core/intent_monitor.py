# python/core/intent_monitor.py
"""
Background monitor for intent lifecycle management.
Handles abandonment detection, periodic decay checks, and nudges.
"""

import threading
import time
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class IntentMonitor(threading.Thread):
    """
    Background thread that monitors intent states and triggers
    periodic maintenance operations.
    """
    
    def __init__(self, 
                 reasoner,
                 check_interval: float = 8.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the IntentMonitor.
        
        Args:
            reasoner: EARLIntentReasoner instance to monitor
            check_interval: Seconds between monitoring checks
            config: Optional configuration dictionary
        """
        super().__init__(daemon=True, name="IntentMonitor")
        self.reasoner = reasoner
        self.check_interval = check_interval
        self.config = config or {}
        
        # Control flags
        self._stop = threading.Event()
        self._pause = threading.Event()
        
        # Task queue for async operations
        self.task_queue = Queue()
        
        # Monitoring statistics
        self.stats = {
            "checks_performed": 0,
            "intents_abandoned": 0,
            "intents_diminished": 0,
            "nudges_sent": 0,
            "last_check": None
        }
        
        # Configuration
        self.abandonment_threshold = self.config.get("abandonment_turns", 10)
        self.nudge_threshold = self.config.get("nudge_turns", 5)
        self.enable_nudges = self.config.get("enable_nudges", True)
        
        # Callback handlers
        self.on_abandonment: Optional[Callable] = None
        self.on_nudge: Optional[Callable] = None
        
        logger.info(f"IntentMonitor initialized with {check_interval}s interval")
    
    def run(self):
        """Main monitoring loop."""
        logger.info("IntentMonitor started")
        
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(1)
                continue
            
            try:
                # Process any queued tasks
                self._process_tasks()
                
                # Perform monitoring checks
                self._perform_checks()
                
                # Update statistics
                self.stats["checks_performed"] += 1
                self.stats["last_check"] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Error in IntentMonitor: {e}", exc_info=True)
            
            # Wait for next check interval
            self._stop.wait(self.check_interval)
        
        logger.info("IntentMonitor stopped")
    
    def _perform_checks(self):
        """Perform all monitoring checks."""
        current_turn = self.reasoner.turn_count
        
        # Check for abandoned intents
        abandoned = self._check_abandonment(current_turn)
        self.stats["intents_abandoned"] += len(abandoned)
        
        # Check for nudge opportunities
        if self.enable_nudges:
            nudged = self._check_nudges(current_turn)
            self.stats["nudges_sent"] += len(nudged)
        
        # Check for confidence decay
        diminished = self._check_decay(current_turn)
        self.stats["intents_diminished"] += len(diminished)
        
        # Log monitoring results
        if abandoned or nudged or diminished:
            logger.info(
                f"Monitor check: {len(abandoned)} abandoned, "
                f"{len(nudged)} nudged, {len(diminished)} diminished"
            )
    
    def _check_abandonment(self, current_turn: int) -> List[str]:
        """
        Check for abandoned intents.
        
        Args:
            current_turn: Current conversation turn
            
        Returns:
            List of abandoned intent IDs
        """
        abandoned = []
        
        for trace in self.reasoner.get_open_traces():
            inactivity = trace.get_inactivity_in_turns(current_turn)
            
            if inactivity > self.abandonment_threshold:
                # Mark as abandoned
                trace.abandon(
                    reason=f"No activity for {inactivity} turns (threshold: {self.abandonment_threshold})"
                )
                
                # Log to memory vault
                if self.reasoner.memory_vault:
                    self.reasoner.memory_vault.log_intent_close(trace)
                
                abandoned.append(trace.intent_id)
                
                # Trigger callback if set
                if self.on_abandonment:
                    self._queue_task(self.on_abandonment, trace)
                
                logger.info(f"Intent {trace.intent_id} ({trace.name}) abandoned after {inactivity} turns")
        
        # Clean up abandoned traces
        self.reasoner.active_traces = [
            t for t in self.reasoner.active_traces if t.is_active()
        ]
        
        return abandoned
    
    def _check_nudges(self, current_turn: int) -> List[str]:
        """
        Check if any intents need nudging.
        
        Args:
            current_turn: Current conversation turn
            
        Returns:
            List of nudged intent IDs
        """
        nudged = []
        
        for trace in self.reasoner.get_open_traces():
            inactivity = trace.get_inactivity_in_turns(current_turn)
            
            # Check if approaching abandonment threshold
            if (self.nudge_threshold < inactivity < self.abandonment_threshold 
                and trace.confidence > 0.5):
                
                nudged.append(trace.intent_id)
                
                # Trigger nudge callback if set
                if self.on_nudge:
                    self._queue_task(self.on_nudge, trace, inactivity)
                
                logger.debug(f"Nudge opportunity for intent {trace.intent_id} ({trace.name})")
        
        return nudged
    
    def _check_decay(self, current_turn: int) -> List[str]:
        """
        Check for intents that should decay.
        
        Args:
            current_turn: Current conversation turn
            
        Returns:
            List of diminished intent IDs
        """
        diminished = []
        
        for trace in self.reasoner.get_open_traces():
            old_confidence = trace.confidence
            
            # Apply passive decay for inactive intents
            if trace.get_inactivity_in_turns(current_turn) > 0:
                trace.confidence *= self.reasoner.decay_rate
                
                # Check if fallen below threshold
                if trace.confidence < self.reasoner.min_confidence:
                    trace.mark_closed(
                        state="diminished",
                        reason=f"Confidence decayed from {old_confidence:.2f} to {trace.confidence:.2f}"
                    )
                    
                    if self.reasoner.memory_vault:
                        self.reasoner.memory_vault.log_intent_close(trace)
                    
                    diminished.append(trace.intent_id)
                    logger.info(f"Intent {trace.intent_id} diminished due to decay")
        
        # Clean up diminished traces
        self.reasoner.active_traces = [
            t for t in self.reasoner.active_traces if t.is_active()
        ]
        
        return diminished
    
    def _process_tasks(self):
        """Process queued tasks."""
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                if callable(task):
                    task()
                elif isinstance(task, tuple) and len(task) >= 2:
                    func, args = task[0], task[1:]
                    func(*args)
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error processing task: {e}")
    
    def _queue_task(self, func: Callable, *args):
        """Queue a task for async execution."""
        self.task_queue.put((func, *args))
    
    def pause(self):
        """Pause monitoring."""
        self._pause.set()
        logger.info("IntentMonitor paused")
    
    def resume(self):
        """Resume monitoring."""
        self._pause.clear()
        logger.info("IntentMonitor resumed")
    
    def stop(self):
        """Stop the monitoring thread."""
        self._stop.set()
        logger.info("IntentMonitor stopping...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return self.stats.copy()
    
    def set_abandonment_handler(self, handler: Callable):
        """
        Set callback for abandoned intents.
        
        Args:
            handler: Function to call with abandoned IntentTrace
        """
        self.on_abandonment = handler
    
    def set_nudge_handler(self, handler: Callable):
        """
        Set callback for nudge opportunities.
        
        Args:
            handler: Function to call with (IntentTrace, inactivity_turns)
        """
        self.on_nudge = handler
    
    def force_check(self):
        """Force an immediate monitoring check."""
        self._queue_task(self._perform_checks)


class IntentNudgeManager:
    """
    Manages nudging strategies for inactive intents.
    """
    
    def __init__(self, output_callback: Optional[Callable] = None):
        """
        Initialize the nudge manager.
        
        Args:
            output_callback: Function to call with nudge messages
        """
        self.output_callback = output_callback or print
        self.nudge_history: Dict[str, List[datetime]] = {}
        
        # Nudge templates
        self.templates = {
            "reminder": [
                "By the way, were you still interested in {intent_name}?",
                "Should we continue with {intent_name}?",
                "Did you want to finish up with {intent_name}?"
            ],
            "clarification": [
                "Just to clarify, are we done with {intent_name}?",
                "Can I mark {intent_name} as complete?",
                "Is there anything else for {intent_name}?"
            ],
            "suggestion": [
                "Would you like help getting back to {intent_name}?",
                "I can help you continue with {intent_name} if you'd like.",
                "Should I provide more information about {intent_name}?"
            ]
        }
    
    def generate_nudge(self, trace, inactivity_turns: int) -> str:
        """
        Generate an appropriate nudge message.
        
        Args:
            trace: IntentTrace object
            inactivity_turns: Number of inactive turns
            
        Returns:
            Nudge message string
        """
        intent_id = trace.intent_id
        
        # Track nudge history
        if intent_id not in self.nudge_history:
            self.nudge_history[intent_id] = []
        
        # Determine nudge type based on context
        if len(self.nudge_history[intent_id]) == 0:
            # First nudge - gentle reminder
            template_type = "reminder"
        elif inactivity_turns > 8:
            # Long inactivity - clarification
            template_type = "clarification"
        else:
            # Moderate inactivity - suggestion
            template_type = "suggestion"
        
        # Select template
        templates = self.templates[template_type]
        template = templates[len(self.nudge_history[intent_id]) % len(templates)]
        
        # Format message
        message = template.format(intent_name=trace.name)
        
        # Record nudge
        self.nudge_history[intent_id].append(datetime.now())
        
        return message
    
    def send_nudge(self, trace, inactivity_turns: int):
        """
        Send a nudge for an inactive intent.
        
        Args:
            trace: IntentTrace object
            inactivity_turns: Number of inactive turns
        """
        message = self.generate_nudge(trace, inactivity_turns)
        
        # Send via callback
        if self.output_callback:
            self.output_callback({
                "type": "nudge",
                "message": message,
                "intent_id": trace.intent_id,
                "intent_name": trace.name,
                "inactivity": inactivity_turns
            })
        
        logger.info(f"Nudge sent for {trace.intent_id}: {message}")
    
    def should_nudge(self, trace, inactivity_turns: int, 
                     max_nudges: int = 3) -> bool:
        """
        Determine if a nudge should be sent.
        
        Args:
            trace: IntentTrace object
            inactivity_turns: Number of inactive turns
            max_nudges: Maximum nudges per intent
            
        Returns:
            True if nudge should be sent
        """
        intent_id = trace.intent_id
        
        # Check nudge count
        if intent_id in self.nudge_history:
            if len(self.nudge_history[intent_id]) >= max_nudges:
                return False
            
            # Check time since last nudge (avoid spam)
            last_nudge = self.nudge_history[intent_id][-1]
            if (datetime.now() - last_nudge).seconds < 60:
                return False
        
        # Check intent confidence
        if trace.confidence < 0.5:
            return False
        
        return True
