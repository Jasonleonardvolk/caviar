# python/tools/analyze_intents.py
"""
Analytics script for analyzing intent closure states and lifecycle patterns.
Provides comprehensive visualization and insights from MemoryVault logs.
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import argparse
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from core.memory_vault import MemoryVault
from core.intent_trace import CLOSURE_STATES


class IntentAnalyzer:
    """
    Comprehensive analytics for intent lifecycle data.
    """
    
    def __init__(self, memory_vault_path: str = "memory_vault"):
        """
        Initialize the analyzer.
        
        Args:
            memory_vault_path: Path to MemoryVault directory
        """
        self.vault_path = Path(memory_vault_path)
        self.memory_vault = MemoryVault(base_dir=memory_vault_path)
        
        # Color codes for terminal output
        self.colors = {
            "open": "\033[92m",        # Green
            "confirmed": "\033[94m",   # Blue
            "diminished": "\033[93m",  # Yellow
            "superseded": "\033[95m",  # Magenta
            "migrated": "\033[96m",    # Cyan
            "satisfied_elsewhere": "\033[97m",  # White
            "abandoned": "\033[91m",   # Red
            "reset": "\033[0m"         # Reset
        }
        
        # ASCII chart characters
        self.bar_char = "█"
        self.partial_chars = ["▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    
    def analyze_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a specific session or the most recent one.
        
        Args:
            session_id: Optional session ID to analyze
            
        Returns:
            Analysis results dictionary
        """
        # Get session traces
        traces = self.memory_vault.get_all_traces(session_id)
        if not traces:
            print(f"No traces found for session {session_id or 'current'}")
            return {}
        
        # Organize by intent_id
        intents = defaultdict(list)
        for trace in traces:
            intent_id = trace.get("intent_id")
            if intent_id:
                intents[intent_id].append(trace)
        
        # Analyze closure states
        closure_states = Counter()
        closure_reasons = defaultdict(list)
        confidence_at_closure = []
        intent_lifespans = []
        
        for intent_id, events in intents.items():
            # Find opening and closing events
            open_event = None
            close_event = None
            
            for event in events:
                if event.get("event") == "intent_opened":
                    open_event = event
                elif event.get("event") == "intent_closed":
                    close_event = event
            
            if close_event:
                state = close_event.get("closure_state", "unknown")
                closure_states[state] += 1
                confidence_at_closure.append(close_event.get("confidence", 0))
                
                # Extract closure reason
                if "closure_history" in close_event:
                    for history in close_event.get("closure_history", []):
                        if history.get("reason"):
                            closure_reasons[state].append(history["reason"])
                
                # Calculate lifespan
                if open_event and close_event:
                    lifespan = close_event.get("turn_opened", 0)
                    age = close_event.get("age_in_turns", 0)
                    if age > 0:
                        intent_lifespans.append(age)
        
        # Find unresolved intents
        unresolved = []
        for intent_id, events in intents.items():
            # Check if intent was never closed
            has_close = any(e.get("event") == "intent_closed" for e in events)
            if not has_close and events:
                last_event = events[-1]
                if last_event.get("closure_state") == "open":
                    unresolved.append({
                        "intent_id": intent_id,
                        "name": last_event.get("name", "Unknown"),
                        "opened_at": last_event.get("opened_at"),
                        "confidence": last_event.get("confidence", 0),
                        "last_active": last_event.get("last_active_turn", 0)
                    })
        
        return {
            "session_id": session_id or "current",
            "total_intents": len(intents),
            "closure_states": dict(closure_states),
            "unresolved_count": len(unresolved),
            "unresolved_intents": unresolved,
            "avg_confidence_at_closure": sum(confidence_at_closure) / len(confidence_at_closure) if confidence_at_closure else 0,
            "avg_lifespan_turns": sum(intent_lifespans) / len(intent_lifespans) if intent_lifespans else 0,
            "closure_reasons": dict(closure_reasons)
        }
    
    def print_closure_distribution(self, analysis: Dict[str, Any]):
        """
        Print closure state distribution with ASCII bar chart.
        
        Args:
            analysis: Analysis results dictionary
        """
        print("\n" + "="*60)
        print("INTENT CLOSURE STATE DISTRIBUTION")
        print("="*60)
        
        closure_states = analysis.get("closure_states", {})
        if not closure_states:
            print("No closed intents found")
            return
        
        # Find max count for scaling
        max_count = max(closure_states.values()) if closure_states else 1
        max_bar_width = 40
        
        # Print each state with bar
        for state in CLOSURE_STATES:
            if state == "open":
                continue  # Skip open state in closure distribution
            
            count = closure_states.get(state, 0)
            percentage = (count / sum(closure_states.values()) * 100) if closure_states else 0
            
            # Calculate bar width
            bar_width = int((count / max_count) * max_bar_width) if max_count > 0 else 0
            
            # Create bar
            bar = self.bar_char * bar_width
            
            # Apply color
            color = self.colors.get(state, self.colors["reset"])
            
            # Print row
            print(f"{color}{state:20}{self.colors['reset']}: {count:3} ({percentage:5.1f}%) {bar}")
        
        print("\n" + "-"*60)
        print(f"Total Closed Intents: {sum(closure_states.values())}")
        print(f"Average Confidence at Closure: {analysis.get('avg_confidence_at_closure', 0):.3f}")
        print(f"Average Lifespan (turns): {analysis.get('avg_lifespan_turns', 0):.1f}")
    
    def print_unresolved_intents(self, analysis: Dict[str, Any]):
        """
        Print unresolved/abandoned intents.
        
        Args:
            analysis: Analysis results dictionary
        """
        unresolved = analysis.get("unresolved_intents", [])
        
        print("\n" + "="*60)
        print("UNRESOLVED/OPEN INTENTS")
        print("="*60)
        
        if not unresolved:
            print("No unresolved intents - all intents properly closed!")
            return
        
        # Prepare table data
        table_data = []
        for intent in unresolved:
            table_data.append([
                intent.get("name", "Unknown")[:30],
                f"{intent.get('confidence', 0):.2f}",
                intent.get("last_active", 0),
                intent.get("opened_at", "Unknown")[:19] if intent.get("opened_at") else "Unknown"
            ])
        
        # Print table
        headers = ["Intent Name", "Confidence", "Last Turn", "Opened At"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print(f"\nTotal Unresolved: {len(unresolved)}")
    
    def print_closure_reasons(self, analysis: Dict[str, Any]):
        """
        Print common closure reasons by state.
        
        Args:
            analysis: Analysis results dictionary
        """
        reasons = analysis.get("closure_reasons", {})
        
        if not reasons:
            return
        
        print("\n" + "="*60)
        print("CLOSURE REASONS BY STATE")
        print("="*60)
        
        for state, reason_list in reasons.items():
            if reason_list:
                color = self.colors.get(state, self.colors["reset"])
                print(f"\n{color}{state.upper()}{self.colors['reset']}:")
                
                # Count and display top reasons
                reason_counts = Counter(reason_list)
                for reason, count in reason_counts.most_common(3):
                    print(f"  - {reason[:60]} ({count}x)")
    
    def print_session_summary(self, analysis: Dict[str, Any]):
        """
        Print overall session summary.
        
        Args:
            analysis: Analysis results dictionary
        """
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        
        summary_data = [
            ["Session ID", analysis.get("session_id", "Unknown")],
            ["Total Intents", analysis.get("total_intents", 0)],
            ["Closed Intents", sum(analysis.get("closure_states", {}).values())],
            ["Unresolved Intents", analysis.get("unresolved_count", 0)],
            ["Avg Confidence", f"{analysis.get('avg_confidence_at_closure', 0):.3f}"],
            ["Avg Lifespan", f"{analysis.get('avg_lifespan_turns', 0):.1f} turns"]
        ]
        
        print(tabulate(summary_data, tablefmt="plain"))
    
    def generate_lifecycle_example(self, session_id: Optional[str] = None):
        """
        Generate and print an example intent lifecycle.
        
        Args:
            session_id: Optional session ID
        """
        traces = self.memory_vault.get_all_traces(session_id)
        
        # Find an intent with complete lifecycle
        intents = defaultdict(list)
        for trace in traces:
            intent_id = trace.get("intent_id")
            if intent_id:
                intents[intent_id].append(trace)
        
        # Find a good example (opened and closed)
        example_intent = None
        for intent_id, events in intents.items():
            has_open = any(e.get("event") == "intent_opened" for e in events)
            has_close = any(e.get("event") == "intent_closed" for e in events)
            if has_open and has_close and len(events) >= 2:
                example_intent = (intent_id, events)
                break
        
        if not example_intent:
            print("\nNo complete intent lifecycle found for example")
            return
        
        intent_id, events = example_intent
        
        print("\n" + "="*60)
        print("INTENT LIFECYCLE EXAMPLE")
        print("="*60)
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.get("timestamp", ""))
        
        print(f"\nIntent ID: {intent_id[:8]}...")
        print(f"Intent Name: {events[0].get('name', 'Unknown')}")
        print("\nLifecycle Events:")
        print("-"*40)
        
        for event in events:
            event_type = event.get("event", "unknown")
            timestamp = event.get("timestamp", "Unknown")[:19]
            confidence = event.get("confidence", 0)
            state = event.get("closure_state", "unknown")
            
            color = self.colors.get(state, self.colors["reset"])
            
            print(f"{timestamp} | {event_type:15} | "
                  f"Confidence: {confidence:.2f} | "
                  f"{color}{state}{self.colors['reset']}")
            
            # Show closure reason if available
            if event_type == "intent_closed" and "closure_history" in event:
                for history in event.get("closure_history", []):
                    if history.get("reason"):
                        print(f"  └─ Reason: {history['reason']}")
    
    def export_report(self, analysis: Dict[str, Any], output_path: str):
        """
        Export analysis report to file.
        
        Args:
            analysis: Analysis results
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "analysis": analysis,
            "summary": {
                "total_intents": analysis.get("total_intents", 0),
                "closure_rate": (1 - analysis.get("unresolved_count", 0) / 
                               max(analysis.get("total_intents", 1), 1)) * 100,
                "most_common_closure": max(analysis.get("closure_states", {}).items(),
                                          key=lambda x: x[1])[0] if analysis.get("closure_states") else "none"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport exported to: {output_path}")
    
    def run_full_analysis(self, session_id: Optional[str] = None,
                         export_path: Optional[str] = None):
        """
        Run complete analysis and display results.
        
        Args:
            session_id: Optional session ID
            export_path: Optional export path for report
        """
        # Run analysis
        analysis = self.analyze_session(session_id)
        
        if not analysis:
            return
        
        # Display results
        self.print_session_summary(analysis)
        self.print_closure_distribution(analysis)
        self.print_unresolved_intents(analysis)
        self.print_closure_reasons(analysis)
        self.generate_lifecycle_example(session_id)
        
        # Export if requested
        if export_path:
            self.export_report(analysis, export_path)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)


def main():
    """Main entry point for the analytics script."""
    parser = argparse.ArgumentParser(
        description="Analyze intent closure states and lifecycle patterns"
    )
    parser.add_argument(
        "--vault",
        default="memory_vault",
        help="Path to MemoryVault directory (default: memory_vault)"
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID to analyze (default: most recent)"
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Export report to specified path"
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List available sessions"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = IntentAnalyzer(args.vault)
    
    # List sessions if requested
    if args.list_sessions:
        traces_dir = Path(args.vault) / "traces"
        if traces_dir.exists():
            sessions = [f.stem for f in traces_dir.glob("*.jsonl")]
            print("\nAvailable Sessions:")
            for session in sorted(sessions):
                print(f"  - {session}")
        else:
            print("No sessions found")
        return
    
    # Run analysis
    analyzer.run_full_analysis(args.session, args.export)


if __name__ == "__main__":
    main()
