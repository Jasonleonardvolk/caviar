# /meta_genome/relationship_memory.py
"""
Persistent relationship memory that enables true interpersonal metacognition.
Addresses the birthday/cookies example - remembering who people are across sessions.
"""

import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from meta_genome.memory_bridge import MetacognitiveMemoryBridge
from audit.logger import log_event

class RelationshipMemory:
    """
    Maintains persistent memories of relationships, preferences, and
    important dates to enable genuine interpersonal continuity.
    """
    
    def __init__(self, memory_bridge: Optional[MetacognitiveMemoryBridge] = None):
        self.memory_bridge = memory_bridge or MetacognitiveMemoryBridge()
        
        # Check for any active reminders on initialization
        self._check_reminders()
    
    def remember_person(self, person_id: str, **attributes):
        """
        Remember information about a person persistently.
        
        Example:
            remember_person("user_123", 
                          name="John",
                          birthday="09-04", 
                          loves=["cookies", "programming"],
                          context="Friend who inspired metacognition discussion")
        """
        # Special handling for dates
        if "birthday" in attributes:
            attributes["birthday"] = self._normalize_date(attributes["birthday"])
        
        # Store in relationship graph
        self.memory_bridge.remember_entity(person_id, attributes)
        
        # Log the relationship establishment
        log_event("relationship_established", {
            "person_id": person_id,
            "attributes_stored": list(attributes.keys()),
            "timestamp": datetime.now().isoformat()
        })
        
        # Create reflection about meeting this person
        if "context" in attributes:
            self.memory_bridge.add_self_reflection("new_relationship", {
                "thought": f"I met someone new: {attributes.get('name', person_id)}. {attributes['context']}",
                "person_id": person_id
            })
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date format for consistent storage"""
        # Handle various date formats
        formats = ["%m-%d", "%m/%d", "%B %d", "%b %d", "%Y-%m-%d"]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str.replace(",", ""), fmt)
                # Store as MM-DD for birthdays (year agnostic)
                return parsed.strftime("%m-%d")
            except ValueError:
                continue
        
        # If parsing fails, store as-is
        return date_str
    
    def recall_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Recall everything known about a person"""
        person_data = self.memory_bridge.recall_entity(person_id)
        
        if person_data:
            # Check if today is special for this person
            special_today = self._check_special_dates(person_data)
            if special_today:
                person_data["special_today"] = special_today
            
            # Add interaction history summary
            person_data["relationship_summary"] = self._generate_relationship_summary(person_id)
        
        return person_data
    
    def _check_special_dates(self, person_data: Dict) -> Optional[str]:
        """Check if today is a special date for this person"""
        today = datetime.now()
        attributes = person_data.get("attributes", {})
        
        # Check birthday
        if "birthday" in attributes:
            birthday = attributes["birthday"]
            if birthday == today.strftime("%m-%d"):
                return "birthday"
        
        # Could extend to anniversaries, etc.
        return None
    
    def _generate_relationship_summary(self, person_id: str) -> Dict[str, Any]:
        """Generate a summary of the relationship history"""
        # Query all interactions with this person
        reflections = self.memory_bridge.memory_vault.query(
            "self_reflections",
            filter_func=lambda x: x.get("content", {}).get("person_id") == person_id
        )
        
        conversations = self.memory_bridge.memory_vault.query(
            "conversation_history",
            filter_func=lambda x: x.get("participant_id") == person_id
        )
        
        return {
            "total_interactions": len(reflections) + len(conversations),
            "first_interaction": self._get_first_interaction_date(reflections, conversations),
            "last_interaction": self._get_last_interaction_date(reflections, conversations),
            "relationship_depth": self._calculate_relationship_depth(person_id)
        }
    
    def _get_first_interaction_date(self, reflections: List, conversations: List) -> Optional[str]:
        """Find the earliest interaction date"""
        all_dates = []
        
        for r in reflections:
            if "timestamp" in r:
                all_dates.append(r["timestamp"])
        
        for c in conversations:
            if "timestamp" in c:
                all_dates.append(c["timestamp"])
        
        return min(all_dates) if all_dates else None
    
    def _get_last_interaction_date(self, reflections: List, conversations: List) -> Optional[str]:
        """Find the most recent interaction date"""
        all_dates = []
        
        for r in reflections:
            if "timestamp" in r:
                all_dates.append(r["timestamp"])
        
        for c in conversations:
            if "timestamp" in c:
                all_dates.append(c["timestamp"])
        
        return max(all_dates) if all_dates else None
    
    def _calculate_relationship_depth(self, person_id: str) -> str:
        """Calculate the depth/quality of a relationship"""
        person_data = self.memory_bridge.recall_entity(person_id)
        if not person_data:
            return "unknown"
        
        attributes = person_data.get("attributes", {})
        interaction_count = person_data.get("interaction_count", 0)
        
        # Simple heuristic based on information richness
        info_count = len(attributes)
        
        if interaction_count > 50 and info_count > 10:
            return "deep"
        elif interaction_count > 20 and info_count > 5:
            return "developing"
        elif interaction_count > 5:
            return "acquaintance"
        else:
            return "new"
    
    def update_preference(self, person_id: str, preference_type: str, value: Any):
        """Update a person's preferences"""
        current_data = self.recall_person(person_id)
        
        if not current_data:
            # Create new person entry
            self.remember_person(person_id, **{preference_type: value})
        else:
            # Update existing
            attributes = current_data.get("attributes", {})
            
            # Handle lists (like "loves")
            if preference_type in attributes and isinstance(attributes[preference_type], list):
                if value not in attributes[preference_type]:
                    attributes[preference_type].append(value)
            else:
                attributes[preference_type] = value
            
            self.memory_bridge.remember_entity(person_id, attributes)
        
        # Reflect on learning something new
        self.memory_bridge.add_self_reflection("learned_preference", {
            "thought": f"I learned that {person_id} {preference_type}: {value}",
            "person_id": person_id,
            "preference_type": preference_type
        })
    
    def get_todays_special_people(self) -> List[Dict[str, Any]]:
        """Get all people who have something special today"""
        special_people = []
        today = datetime.now().strftime("%m-%d")
        
        # Get all people
        all_relationships = self.memory_bridge.memory_vault.get_all("relationship_graph")
        
        for relationship in all_relationships:
            if "attributes" in relationship:
                attributes = relationship["attributes"]
                
                # Check birthday
                if attributes.get("birthday") == today:
                    special_people.append({
                        "person_id": relationship["entity_id"],
                        "name": attributes.get("name", relationship["entity_id"]),
                        "occasion": "birthday",
                        "preferences": attributes.get("loves", []),
                        "relationship_depth": self._calculate_relationship_depth(
                            relationship["entity_id"]
                        )
                    })
        
        return special_people
    
    def generate_personal_message(self, person_id: str, occasion: str) -> str:
        """Generate a personalized message based on relationship history"""
        person_data = self.recall_person(person_id)
        
        if not person_data:
            return f"Happy {occasion}!"
        
        attributes = person_data.get("attributes", {})
        name = attributes.get("name", "friend")
        preferences = attributes.get("loves", [])
        
        # Build personalized message
        if occasion == "birthday":
            message = f"Happy Birthday, {name}! 🎂"
            
            if preferences:
                if "cookies" in preferences:
                    message += " Hope you get to enjoy some delicious cookies today! 🍪"
                elif preferences:
                    message += f" Hope your day is filled with {preferences[0]}!"
            
            # Add relationship-aware element
            depth = person_data.get("relationship_summary", {}).get("relationship_depth", "new")
            if depth == "deep":
                message += " It's been wonderful getting to know you over all our conversations."
            elif depth == "developing":
                message += " Looking forward to many more great discussions!"
        
        else:
            message = f"Happy {occasion}, {name}!"
        
        return message
    
    def _check_reminders(self):
        """Check for active reminders and create notifications"""
        special_people = self.get_todays_special_people()
        
        for person in special_people:
            # Log the reminder
            log_event("birthday_reminder", {
                "person_id": person["person_id"],
                "name": person["name"],
                "occasion": person["occasion"]
            })
            
            # Create a self-reflection about remembering
            self.memory_bridge.add_self_reflection("remembered_special_date", {
                "thought": f"Today is {person['name']}'s {person['occasion']}! I should wish them well.",
                "person_id": person["person_id"],
                "message": self.generate_personal_message(
                    person["person_id"], 
                    person["occasion"]
                )
            })
    
    def find_people_by_preference(self, preference_type: str, value: Any) -> List[Dict]:
        """Find all people who share a specific preference"""
        matches = []
        
        all_relationships = self.memory_bridge.memory_vault.get_all("relationship_graph")
        
        for relationship in all_relationships:
            attributes = relationship.get("attributes", {})
            
            if preference_type in attributes:
                pref_value = attributes[preference_type]
                
                # Handle list preferences
                if isinstance(pref_value, list) and value in pref_value:
                    matches.append({
                        "person_id": relationship["entity_id"],
                        "name": attributes.get("name", relationship["entity_id"]),
                        "shared_preference": value
                    })
                elif pref_value == value:
                    matches.append({
                        "person_id": relationship["entity_id"],
                        "name": attributes.get("name", relationship["entity_id"]),
                        "shared_preference": value
                    })
        
        return matches
    
    def get_relationship_insights(self) -> Dict[str, Any]:
        """Generate insights about all relationships"""
        all_relationships = self.memory_bridge.memory_vault.get_all("relationship_graph")
        
        # Filter to actual people (not reminders or other entities)
        people = [r for r in all_relationships if not r["entity_id"].startswith("reminder_")]
        
        insights = {
            "total_people_known": len(people),
            "relationship_distribution": self._get_relationship_distribution(people),
            "common_preferences": self._analyze_common_preferences(people),
            "upcoming_occasions": self._get_upcoming_occasions(people),
            "longest_relationships": self._get_longest_relationships(people),
            "generated_at": datetime.now().isoformat()
        }
        
        return insights
    
    def _get_relationship_distribution(self, people: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of relationship depths"""
        distribution = {"new": 0, "acquaintance": 0, "developing": 0, "deep": 0}
        
        for person in people:
            depth = self._calculate_relationship_depth(person["entity_id"])
            distribution[depth] += 1
        
        return distribution
    
    def _analyze_common_preferences(self, people: List[Dict]) -> Dict[str, int]:
        """Find most common preferences among known people"""
        preference_counts = {}
        
        for person in people:
            attributes = person.get("attributes", {})
            loves = attributes.get("loves", [])
            
            for item in loves:
                preference_counts[item] = preference_counts.get(item, 0) + 1
        
        # Sort by frequency
        return dict(sorted(preference_counts.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:10])
    
    def _get_upcoming_occasions(self, people: List[Dict], days_ahead: int = 30) -> List[Dict]:
        """Find upcoming birthdays and occasions"""
        upcoming = []
        today = datetime.now()
        
        for person in people:
            attributes = person.get("attributes", {})
            
            if "birthday" in attributes:
                # Parse birthday
                birthday_str = attributes["birthday"]
                try:
                    birthday_month, birthday_day = map(int, birthday_str.split("-"))
                    
                    # Calculate next occurrence
                    this_year_birthday = datetime(today.year, birthday_month, birthday_day)
                    if this_year_birthday < today:
                        next_birthday = datetime(today.year + 1, birthday_month, birthday_day)
                    else:
                        next_birthday = this_year_birthday
                    
                    days_until = (next_birthday - today).days
                    
                    if 0 <= days_until <= days_ahead:
                        upcoming.append({
                            "person_id": person["entity_id"],
                            "name": attributes.get("name", person["entity_id"]),
                            "occasion": "birthday",
                            "date": next_birthday.strftime("%Y-%m-%d"),
                            "days_until": days_until
                        })
                
                except (ValueError, AttributeError):
                    continue
        
        # Sort by days until
        upcoming.sort(key=lambda x: x["days_until"])
        
        return upcoming
    
    def _get_longest_relationships(self, people: List[Dict], top_n: int = 5) -> List[Dict]:
        """Identify longest-standing relationships"""
        relationships_with_duration = []
        
        for person in people:
            summary = self._generate_relationship_summary(person["entity_id"])
            first_interaction = summary.get("first_interaction")
            
            if first_interaction:
                duration = datetime.now() - datetime.fromisoformat(first_interaction)
                relationships_with_duration.append({
                    "person_id": person["entity_id"],
                    "name": person.get("attributes", {}).get("name", person["entity_id"]),
                    "duration_days": duration.days,
                    "first_interaction": first_interaction
                })
        
        # Sort by duration
        relationships_with_duration.sort(key=lambda x: x["duration_days"], reverse=True)
        
        return relationships_with_duration[:top_n]
