"""narrative_recap.py - Session and concept history narration system.

This module provides the NarrativeRecap system that analyzes concept events
and generates human-readable summaries of ALAN's cognitive activity. It:

1. Reconstructs cognitive sessions from event logs
2. Summarizes concept stability changes, merges, and pruning
3. Identifies significant phase coherence and resonance events
4. Generates natural language descriptions of cognitive activity

The system serves as ALAN's "autobiographical memory," allowing it to
narrate its own cognitive evolution and maintain a sense of continuity
across sessions.
"""

import logging
import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from collections import defaultdict, Counter
import math

try:
    # Try absolute import first
    from concept_logger import ConceptLogger, default_concept_logger
except ImportError:
    # Fallback to relative import
    from .concept_logger import ConceptLogger, default_concept_logger
try:
    # Try absolute import first
    from concept_metadata import ConceptMetadata 
except ImportError:
    # Fallback to relative import
    from .concept_metadata import ConceptMetadata 
try:
    # Try absolute import first
    from time_context import TimeContext, default_time_context
except ImportError:
    # Fallback to relative import
    from .time_context import TimeContext, default_time_context

# Configure logger
logger = logging.getLogger("narrative_recap")

class EventSummary:
    """Container for summarized concept event statistics."""
    
    def __init__(self):
        """Initialize empty event summary."""
        # New concepts
        self.new_concepts: List[Dict[str, Any]] = []
        
        # Merged concepts
        self.merged_concepts: List[Dict[str, Any]] = []
        
        # Pruned concepts
        self.pruned_concepts: List[Dict[str, Any]] = []
        
        # Stability changes
        self.stability_increases: List[Dict[str, Any]] = []
        self.stability_decreases: List[Dict[str, Any]] = []
        
        # Phase events
        self.phase_events: List[Dict[str, Any]] = []
        
        # Activations
        self.top_activations: List[Dict[str, Any]] = []
        
        # Errors/warnings
        self.errors: List[Dict[str, Any]] = []
        
    def get_stats(self) -> Dict[str, int]:
        """Get statistical summary of events."""
        return {
            "new_concepts": len(self.new_concepts),
            "merged_concepts": len(self.merged_concepts),
            "pruned_concepts": len(self.pruned_concepts),
            "stability_increases": len(self.stability_increases),
            "stability_decreases": len(self.stability_decreases),
            "phase_events": len(self.phase_events),
            "top_activations": len(self.top_activations),
            "errors": len(self.errors),
            "total_events": sum([
                len(self.new_concepts),
                len(self.merged_concepts),
                len(self.pruned_concepts),
                len(self.stability_increases),
                len(self.stability_decreases),
                len(self.phase_events),
                len(self.top_activations),
                len(self.errors)
            ])
        }
        
    def has_significant_events(self) -> bool:
        """Check if summary contains significant events."""
        return (len(self.new_concepts) > 0 or
                len(self.merged_concepts) > 0 or
                len(self.pruned_concepts) > 0 or
                len(self.stability_increases) > 0 or
                len(self.phase_events) > 0)


class NarrativeRecap:
    """
    Session and concept history narration system.
    
    NarrativeRecap analyzes concept events and generates human-readable
    summaries of ALAN's cognitive activity, forming a system for
    self-reflection and autobiographical memory.
    
    Attributes:
        log_file: Path to concept event log file
        time_context: TimeContext for temporal references
        max_events: Maximum events to process per recap
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        time_context: Optional[TimeContext] = None,
        max_events: int = 1000
    ):
        """
        Initialize the NarrativeRecap system.
        
        Args:
            log_file: Path to concept event log file (uses default logger if None)
            time_context: TimeContext for temporal references (uses default if None)
            max_events: Maximum events to process per recap
        """
        self.log_file = log_file or default_concept_logger.log_file
        self.time_context = time_context or default_time_context
        self.max_events = max_events
        
        # Keep track of last processed timestamp
        self.last_processed_time: Optional[datetime] = None
        
        # Mapping of concept IDs to names (for reference)
        self.concept_names: Dict[str, str] = {}
        
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a log line into a structured event.
        
        Args:
            line: Raw log line from the event log
            
        Returns:
            Dict containing parsed event or None if parsing failed
        """
        # Skip empty lines
        if not line.strip():
            return None
            
        # Check if the log is in JSON format
        if line.strip().startswith('{') and line.strip().endswith('}'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
        
        # Parse standard log format: timestamp | level | event_type | id | message | details...
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| (\w+) \| ([A-Z_]+) \| ([^ |]+) \| ([^|]+)(?:\s\|\s(.+))?'
        match = re.match(pattern, line)
        
        if not match:
            return None
            
        timestamp_str, level, event_type, subject_id, message, details_str = match.groups()
        
        # Parse timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                return None
        
        # Parse details into dictionary
        details = {}
        if details_str:
            for detail in details_str.split(' | '):
                if '=' in detail:
                    key, value = detail.split('=', 1)
                    details[key.strip()] = value.strip()
        
        # Create structured event
        event = {
            "timestamp": timestamp,
            "level": level,
            "event_type": event_type,
            "subject_id": subject_id,
            "message": message.strip(),
            "details": details
        }
        
        return event
    
    def extract_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract events from log file within a time range.
        
        Args:
            start_time: Start of time range (None for beginning of log)
            end_time: End of time range (None for current time)
            
        Returns:
            List of parsed events
        """
        if not os.path.exists(self.log_file):
            logger.warning(f"Log file not found: {self.log_file}")
            return []
            
        events = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    event = self.parse_log_line(line)
                    if not event:
                        continue
                        
                    # Check if event is within time range
                    timestamp = event["timestamp"]
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                        
                    # Track concept names for reference
                    if event["event_type"] in ["BIRTH", "ACTIVATION"]:
                        if "name" in event["details"]:
                            concept_id = event["subject_id"]
                            self.concept_names[concept_id] = event["details"]["name"]
                    
                    events.append(event)
                    
                    # Limit number of events
                    if len(events) >= self.max_events:
                        logger.warning(f"Reached maximum events limit ({self.max_events})")
                        break
        except Exception as e:
            logger.error(f"Error reading log file: {str(e)}")
            
        return events
    
    def summarize_events(
        self,
        events: List[Dict[str, Any]],
        min_significance: float = 0.1
    ) -> EventSummary:
        """
        Summarize concept events into categories.
        
        Args:
            events: List of parsed events
            min_significance: Minimum change significance threshold
            
        Returns:
            EventSummary object containing categorized events
        """
        summary = EventSummary()
        
        # Activations by concept (to find top activations)
        activations = defaultdict(list)
        
        for event in events:
            event_type = event["event_type"]
            subject_id = event["subject_id"]
            
            # Get concept name if available
            concept_name = None
            if subject_id in self.concept_names:
                concept_name = self.concept_names[subject_id]
            elif "name" in event["details"]:
                concept_name = event["details"]["name"]
                self.concept_names[subject_id] = concept_name
                
            # Process by event type
            if event_type == "BIRTH":
                summary.new_concepts.append({
                    "concept_id": subject_id,
                    "name": concept_name,
                    "timestamp": event["timestamp"],
                    "source": event["details"].get("source", "unknown")
                })
                
            elif event_type == "MERGE":
                # Collect parent names for better narrative
                parent_names = []
                if "parent_names" in event["details"]:
                    if isinstance(event["details"]["parent_names"], list):
                        parent_names = event["details"]["parent_names"]
                    else:
                        # Try parsing from string
                        try:
                            parent_names = event["details"]["parent_names"].strip('[]').split(', ')
                        except:
                            pass
                
                summary.merged_concepts.append({
                    "concept_id": subject_id,
                    "name": concept_name or event["details"].get("child_name", "unknown"),
                    "timestamp": event["timestamp"],
                    "reason": event["details"].get("reason", "unknown"),
                    "parent_names": parent_names,
                    "score": float(event["details"].get("score", 0)) if "score" in event["details"] else None
                })
                
            elif event_type == "PRUNE":
                summary.pruned_concepts.append({
                    "concept_id": subject_id,
                    "name": concept_name,
                    "timestamp": event["timestamp"],
                    "reason": event["details"].get("reason", "unknown"),
                    "metrics": {k: v for k, v in event["details"].items() if k not in ["name", "reason"]}
                })
                
            elif event_type == "STABILITY":
                # Parse old and new values
                old_val = float(event["details"].get("old", 0))
                new_val = float(event["details"].get("new", 0))
                change = new_val - old_val
                
                # Skip minor changes
                if abs(change) < min_significance:
                    continue
                    
                stability_event = {
                    "concept_id": subject_id,
                    "name": concept_name,
                    "timestamp": event["timestamp"],
                    "old_value": old_val,
                    "new_value": new_val,
                    "change": change
                }
                
                if change > 0:
                    summary.stability_increases.append(stability_event)
                else:
                    summary.stability_decreases.append(stability_event)
                    
            elif event_type == "PHASE":
                summary.phase_events.append({
                    "concept_ids": subject_id.split(','),  # May contain multiple IDs
                    "timestamp": event["timestamp"],
                    "coherence": float(event["details"].get("coherence", 0)),
                    "event": event["details"].get("event", "unknown"),
                    "concept_names": event["details"].get("names", [])
                })
                
            elif event_type == "ACTIVATION":
                # Track activations for later analysis
                strength = float(event["details"].get("strength", 0))
                activations[subject_id].append({
                    "timestamp": event["timestamp"],
                    "strength": strength,
                    "name": concept_name,
                    "context": event["details"].get("context", None)
                })
                
            elif event_type == "ERROR":
                summary.errors.append({
                    "concept_id": subject_id,
                    "timestamp": event["timestamp"],
                    "operation": event["details"].get("operation", "unknown"),
                    "severity": event["details"].get("severity", "WARNING"),
                    "message": event["message"]
                })
        
        # Find top activations (by frequency and strength)
        for concept_id, acts in activations.items():
            if len(acts) < 2:  # Skip concepts with very few activations
                continue
                
            avg_strength = sum(a["strength"] for a in acts) / len(acts)
            max_strength = max(a["strength"] for a in acts)
            
            summary.top_activations.append({
                "concept_id": concept_id,
                "name": acts[0]["name"],  # Use name from first activation
                "count": len(acts),
                "avg_strength": avg_strength,
                "max_strength": max_strength,
                "first_timestamp": min(a["timestamp"] for a in acts),
                "last_timestamp": max(a["timestamp"] for a in acts)
            })
            
        # Sort top activations by count (descending)
        summary.top_activations.sort(key=lambda x: x["count"], reverse=True)
        
        return summary
    
    def generate_recap(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format_type: str = "plain"
    ) -> str:
        """
        Generate a narrative recap of concept events.
        
        Args:
            start_time: Start of time range (None for since last recap)
            end_time: End of time range (None for current time)
            format_type: Output format ('plain', 'markdown', 'html')
            
        Returns:
            Formatted narrative recap
        """
        # Default time range: since last recap or last 24 hours
        if not start_time:
            if self.last_processed_time:
                start_time = self.last_processed_time
            else:
                start_time = datetime.now() - timedelta(days=1)
                
        if not end_time:
            end_time = datetime.now()
            
        # Extract events in time range
        events = self.extract_events(start_time, end_time)
        
        if not events:
            self.last_processed_time = end_time
            return "No concept events found in the specified time period."
            
        # Summarize events
        summary = self.summarize_events(events)
        
        if not summary.has_significant_events():
            self.last_processed_time = end_time
            return "No significant concept activity in the specified time period."
            
        # Generate narrative text
        narrative = self._format_narrative(summary, start_time, end_time, format_type)
        
        # Update last processed time
        self.last_processed_time = end_time
        
        return narrative
    
    def _format_narrative(
        self,
        summary: EventSummary,
        start_time: datetime,
        end_time: datetime,
        format_type: str
    ) -> str:
        """
        Format event summary into narrative text.
        
        Args:
            summary: Summarized events
            start_time: Start of time range
            end_time: End of time range
            format_type: Output format
            
        Returns:
            Formatted narrative text
        """
        stats = summary.get_stats()
        
        # Time period description
        time_diff = end_time - start_time
        if time_diff.days > 0:
            period = f"{time_diff.days} days"
        elif time_diff.seconds >= 3600:
            period = f"{time_diff.seconds // 3600} hours"
        else:
            period = f"{time_diff.seconds // 60} minutes"
            
        # Markdown heading markers
        h1, h2, bold_start, bold_end = "", "", "", ""
        if format_type == "markdown":
            h1, h2 = "# ", "## "
            bold_start, bold_end = "**", "**"
        elif format_type == "html":
            h1, h2 = "<h1>", "</h1>"
            bold_start, bold_end = "<b>", "</b>"
            
        # Generate header
        lines = []
        lines.append(f"{h1}Cognitive Session Recap{h1}")
        lines.append("")
        lines.append(f"Time period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} ({period})")
        lines.append("")
        
        # Summary section
        lines.append(f"{h2}Summary{h2}")
        lines.append("")
        
        # Generate concise summary
        summary_items = []
        
        if stats["new_concepts"] > 0:
            summary_items.append(f"stabilized {stats['new_concepts']} new concepts")
            
        if stats["merged_concepts"] > 0:
            merged_txt = f"merged {stats['merged_concepts']} concept"
            if stats["merged_concepts"] > 1:
                merged_txt += "s"
            summary_items.append(merged_txt)
            
        if stats["pruned_concepts"] > 0:
            summary_items.append(f"pruned {stats['pruned_concepts']} concepts")
            
        if stats["stability_increases"] > 0:
            summary_items.append(f"strengthened {stats['stability_increases']} concepts")
            
        if stats["stability_decreases"] > 0:
            summary_items.append(f"weakened {stats['stability_decreases']} concepts")
            
        if stats["phase_events"] > 0:
            summary_items.append(f"experienced {stats['phase_events']} phase events")
            
        # Construct the summary sentence
        if summary_items:
            sentence = "During this period, I " + ", ".join(summary_items) + "."
            lines.append(sentence)
            lines.append("")
            
        # Detail sections
        if summary.new_concepts:
            lines.append(f"{h2}New Concepts{h2}")
            lines.append("")
            for concept in summary.new_concepts:
                name = concept["name"] or concept["concept_id"]
                lines.append(f"- {bold_start}{name}{bold_end} from source: {concept['source']}")
            lines.append("")
                
        if summary.merged_concepts:
            lines.append(f"{h2}Merged Concepts{h2}")
            lines.append("")
            for merge in summary.merged_concepts:
                name = merge["name"] or merge["concept_id"]
                parent_str = ""
                if merge["parent_names"]:
                    if len(merge["parent_names"]) == 1:
                        parent_str = f" from {merge['parent_names'][0]}"
                    elif len(merge["parent_names"]) == 2:
                        parent_str = f" from {merge['parent_names'][0]} and {merge['parent_names'][1]}"
                    else:
                        parent_str = f" from {len(merge['parent_names'])} parent concepts"
                        
                reason = f" due to {merge['reason']}" if merge["reason"] != "unknown" else ""
                score_str = f" (score: {merge['score']:.2f})" if merge["score"] is not None else ""
                
                lines.append(f"- Created {bold_start}{name}{bold_end}{parent_str}{reason}{score_str}")
            lines.append("")
                
        if summary.pruned_concepts:
            lines.append(f"{h2}Pruned Concepts{h2}")
            lines.append("")
            for prune in summary.pruned_concepts:
                name = prune["name"] or prune["concept_id"]
                reason = f" due to {prune['reason']}" if prune["reason"] != "unknown" else ""
                
                # Include relevant metrics if available
                metrics_str = ""
                for key in ["entropy", "stability", "age"]:
                    if key in prune["metrics"]:
                        metrics_str += f", {key}: {prune['metrics'][key]}"
                
                lines.append(f"- Removed {bold_start}{name}{bold_end}{reason}{metrics_str}")
            lines.append("")
            
        if summary.phase_events:
            lines.append(f"{h2}Significant Phase Events{h2}")
            lines.append("")
            for event in summary.phase_events:
                event_desc = event["event"]
                coherence = event["coherence"]
                
                # Use concept names if available
                concept_desc = ""
                if event["concept_names"]:
                    if len(event["concept_names"]) == 1:
                        concept_desc = f" in {event['concept_names'][0]}"
                    elif len(event["concept_names"]) == 2:
                        concept_desc = f" between {event['concept_names'][0]} and {event['concept_names'][1]}"
                    else:
                        concept_desc = f" across {len(event['concept_names'])} concepts"
                        
                lines.append(f"- {bold_start}{event_desc}{bold_end}{concept_desc} (coherence: {coherence:.2f})")
            lines.append("")
            
        if summary.top_activations and len(summary.top_activations) > 0:
            lines.append(f"{h2}Most Active Concepts{h2}")
            lines.append("")
            # Only show top 5
            for activation in summary.top_activations[:5]:
                name = activation["name"] or activation["concept_id"]
                count = activation["count"]
                avg_strength = activation["avg_strength"]
                
                lines.append(f"- {bold_start}{name}{bold_end} activated {count} times (avg strength: {avg_strength:.2f})")
            lines.append("")
            
        if summary.errors:
            lines.append(f"{h2}Issues{h2}")
            lines.append("")
            # Group errors by type
            error_types = Counter()
            for error in summary.errors:
                op = error["operation"]
                error_types[op] += 1
                
            for op, count in error_types.most_common():
                if count == 1:
                    lines.append(f"- Single issue in {op} operation")
                else:
                    lines.append(f"- {count} issues in {op} operation")
            lines.append("")
            
        # Format based on output type
        if format_type == "html":
            return "<p>" + "</p><p>".join([l if l else "&nbsp;" for l in lines]) + "</p>"
        else:
            return "\n".join(lines)
            
    def get_daily_recap(self, format_type: str = "plain") -> str:
        """
        Generate a daily recap for the last 24 hours.
        
        Args:
            format_type: Output format ('plain', 'markdown', 'html')
            
        Returns:
            Formatted daily recap
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        return self.generate_recap(start_time, end_time, format_type)
        
    def get_session_recap(
        self,
        session_start: Optional[datetime] = None,
        format_type: str = "plain"
    ) -> str:
        """
        Generate a recap for the current session.
        
        Args:
            session_start: Session start time (None for 6 hours ago)
            format_type: Output format ('plain', 'markdown', 'html')
            
        Returns:
            Formatted session recap
        """
        if not session_start:
            session_start = datetime.now() - timedelta(hours=6)
            
        return self.generate_recap(session_start, None, format_type)

# For convenience, create a singleton instance
default_narrative_recap = NarrativeRecap()
