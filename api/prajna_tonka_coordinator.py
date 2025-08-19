"""
Prajna-TONKA Coordinator
Allows Prajna to interpret complex requests and delegate code tasks to TONKA
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TonkaTaskType(Enum):
    """Types of tasks TONKA can handle"""
    CODE_GENERATION = "code_generation"
    PROJECT_CREATION = "project_creation"
    ALGORITHM_IMPLEMENTATION = "algorithm_implementation"
    CODE_EXPLANATION = "code_explanation"
    CODE_REFACTORING = "code_refactoring"
    DEBUG_ASSISTANCE = "debug_assistance"

@dataclass
class TonkaOrder:
    """Represents an order from Prajna to TONKA"""
    task_type: TonkaTaskType
    description: str
    language: str = "python"
    style: str = "clean"
    context: Dict[str, Any] = None
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "description": self.description,
            "language": self.language,
            "style": self.style,
            "context": self.context or {},
            "priority": self.priority
        }

class PrajnaTonkaCoordinator:
    """
    Coordinates between Prajna's language understanding and TONKA's code generation
    Prajna can analyze complex requests and break them down into TONKA orders
    """
    
    def __init__(self):
        self.command_patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> List[Tuple[re.Pattern, TonkaTaskType, str]]:
        """Compile regex patterns for command recognition"""
        patterns = [
            # Direct TONKA commands
            (r"(?:tonka|@tonka|hey tonka)[,:]?\s+(.+)", TonkaTaskType.CODE_GENERATION, "direct"),
            
            # Code generation patterns
            (r"(?:create|write|generate|build|make)\s+(?:me\s+)?(?:a\s+)?(.+?)(?:\s+in\s+(\w+))?", TonkaTaskType.CODE_GENERATION, "create"),
            (r"(?:implement|code|program)\s+(.+?)(?:\s+using\s+(\w+))?", TonkaTaskType.CODE_GENERATION, "implement"),
            (r"show me (?:the\s+)?code for\s+(.+)", TonkaTaskType.CODE_GENERATION, "show"),
            (r"i need (?:a\s+)?(?:function|class|method)\s+(?:to\s+|that\s+|for\s+)(.+)", TonkaTaskType.CODE_GENERATION, "need"),
            
            # Project creation patterns
            (r"(?:scaffold|bootstrap|create)\s+(?:a\s+)?(\w+)\s+project\s+(?:called\s+)?([^\s]+)", TonkaTaskType.PROJECT_CREATION, "scaffold"),
            (r"new\s+(\w+)\s+project\s+([^\s]+)", TonkaTaskType.PROJECT_CREATION, "new_project"),
            (r"set up (?:a\s+)?(.+?)\s+project", TonkaTaskType.PROJECT_CREATION, "setup"),
            
            # Algorithm patterns
            (r"(?:implement\s+)?(\w+)\s+algorithm", TonkaTaskType.ALGORITHM_IMPLEMENTATION, "algorithm"),
            (r"show me (?:the\s+)?(\w+)\s+algorithm", TonkaTaskType.ALGORITHM_IMPLEMENTATION, "show_algorithm"),
            (r"how (?:do you|to)\s+implement\s+(\w+)", TonkaTaskType.ALGORITHM_IMPLEMENTATION, "how_implement"),
            
            # Code explanation patterns
            (r"explain this code[:\s]*(.+)", TonkaTaskType.CODE_EXPLANATION, "explain"),
            (r"what does this (?:code|function|class) do[:\s]*(.+)", TonkaTaskType.CODE_EXPLANATION, "what_does"),
            (r"help me understand[:\s]*(.+)", TonkaTaskType.CODE_EXPLANATION, "understand"),
            
            # Refactoring patterns
            (r"(?:refactor|improve|optimize|clean up)\s+(?:this\s+)?(?:code)?[:\s]*(.+)", TonkaTaskType.CODE_REFACTORING, "refactor"),
            (r"make this (?:code\s+)?(?:better|cleaner|faster)[:\s]*(.+)", TonkaTaskType.CODE_REFACTORING, "improve"),
            
            # Debug patterns
            (r"(?:debug|fix|solve)\s+(?:this\s+)?(?:error|bug|issue)?[:\s]*(.+)", TonkaTaskType.DEBUG_ASSISTANCE, "debug"),
            (r"why (?:is this|does this)\s+(?:not working|failing|broken)[:\s]*(.+)", TonkaTaskType.DEBUG_ASSISTANCE, "why_broken"),
        ]
        
        return [(re.compile(pattern, re.IGNORECASE), task_type, label) 
                for pattern, task_type, label in patterns]
    
    def analyze_request(self, user_query: str) -> Optional[TonkaOrder]:
        """
        Analyze user query to determine if it contains TONKA orders
        Returns None if no code-related request is found
        """
        query_lower = user_query.lower().strip()
        
        # Check each pattern
        for pattern, task_type, label in self.command_patterns:
            match = pattern.search(user_query)
            if match:
                logger.info(f"🎯 Matched pattern '{label}' for task type: {task_type.value}")
                
                # Extract the core request
                groups = match.groups()
                description = groups[0] if groups else user_query
                
                # Extract language if specified
                language = "python"  # default
                if len(groups) > 1 and groups[1]:
                    language = self._normalize_language(groups[1])
                
                # Determine style based on context
                style = self._determine_style(query_lower, task_type)
                
                # Extract any code context
                context = self._extract_context(user_query, match)
                
                return TonkaOrder(
                    task_type=task_type,
                    description=description.strip(),
                    language=language,
                    style=style,
                    context=context,
                    priority=self._calculate_priority(query_lower)
                )
        
        # Check for implicit code requests
        if self._is_implicit_code_request(query_lower):
            return TonkaOrder(
                task_type=TonkaTaskType.CODE_GENERATION,
                description=user_query,
                language="python",
                style="simple"
            )
        
        return None
    
    def _normalize_language(self, lang: str) -> str:
        """Normalize language names"""
        lang = lang.lower().strip()
        
        language_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "cpp": "c++",
            "c#": "csharp",
            "golang": "go",
            "rb": "ruby",
            "rs": "rust"
        }
        
        return language_map.get(lang, lang)
    
    def _determine_style(self, query: str, task_type: TonkaTaskType) -> str:
        """Determine coding style from query context"""
        if any(word in query for word in ["production", "professional", "robust"]):
            return "production"
        elif any(word in query for word in ["simple", "basic", "example"]):
            return "simple"
        elif any(word in query for word in ["optimize", "fast", "performance"]):
            return "optimized"
        elif task_type == TonkaTaskType.PROJECT_CREATION:
            return "scaffold"
        else:
            return "clean"
    
    def _extract_context(self, query: str, match: re.Match) -> Dict[str, Any]:
        """Extract additional context from the query"""
        context = {}
        
        # Check for code blocks
        code_block_pattern = r'```(?:\w+)?\n(.*?)```'
        code_matches = re.findall(code_block_pattern, query, re.DOTALL)
        if code_matches:
            context["code_samples"] = code_matches
        
        # Check for specific requirements
        if "async" in query.lower():
            context["async"] = True
        if "test" in query.lower() or "testing" in query.lower():
            context["include_tests"] = True
        if "type" in query.lower() and ("hint" in query.lower() or "annotation" in query.lower()):
            context["type_hints"] = True
        
        # Extract any mentioned features for projects
        if "features:" in query.lower():
            features_match = re.search(r'features:\s*([^\n]+)', query, re.IGNORECASE)
            if features_match:
                features = [f.strip() for f in features_match.group(1).split(',')]
                context["features"] = features
        
        return context
    
    def _calculate_priority(self, query: str) -> int:
        """Calculate task priority based on urgency indicators"""
        if any(word in query for word in ["urgent", "asap", "immediately", "now"]):
            return 3
        elif any(word in query for word in ["please", "when you can", "later"]):
            return 1
        else:
            return 2
    
    def _is_implicit_code_request(self, query: str) -> bool:
        """Check if query is implicitly asking for code"""
        code_indicators = [
            "how do i", "how to", "how can i",
            "example of", "sample",
            "function", "class", "method",
            "algorithm", "data structure",
            "api", "endpoint", "route",
            "parse", "validate", "convert",
            "connect to", "integrate with"
        ]
        
        return any(indicator in query for indicator in code_indicators)
    
    def extract_multiple_orders(self, user_query: str) -> List[TonkaOrder]:
        """
        Extract multiple TONKA orders from a complex request
        E.g., "Create an API project and implement user authentication"
        """
        orders = []
        
        # Split by conjunctions
        parts = re.split(r'\s+(?:and|then|also|plus)\s+', user_query, flags=re.IGNORECASE)
        
        for part in parts:
            order = self.analyze_request(part.strip())
            if order:
                orders.append(order)
        
        # If no parts found, try single analysis
        if not orders:
            single_order = self.analyze_request(user_query)
            if single_order:
                orders.append(single_order)
        
        return orders
    
    def create_tonka_request(self, order: TonkaOrder) -> Dict[str, Any]:
        """
        Convert a TonkaOrder into a request for TONKA API
        """
        if order.task_type == TonkaTaskType.CODE_GENERATION:
            return {
                "task": order.description,
                "language": order.language,
                "style": order.style,
                "context": order.context
            }
        
        elif order.task_type == TonkaTaskType.PROJECT_CREATION:
            # Parse project type and name
            parts = order.description.split()
            project_type = "basic"
            project_name = "my_project"
            
            # Common project types
            if any(t in order.description.lower() for t in ["api", "rest", "fastapi"]):
                project_type = "api"
            elif any(t in order.description.lower() for t in ["cli", "command"]):
                project_type = "cli"
            
            # Extract name (last word usually)
            words = order.description.split()
            if words:
                project_name = words[-1].replace(".", "").replace(",", "")
            
            return {
                "project_type": project_type,
                "name": project_name,
                "features": order.context.get("features", [])
            }
        
        elif order.task_type == TonkaTaskType.ALGORITHM_IMPLEMENTATION:
            return {
                "algorithm": order.description,
                "language": order.language
            }
        
        else:
            # Default to code generation
            return {
                "task": order.description,
                "language": order.language,
                "style": order.style
            }
    
    def format_response_with_context(self, tonka_response: Dict[str, Any], 
                                   order: TonkaOrder, 
                                   prajna_context: str = "") -> str:
        """
        Format TONKA's response with Prajna's contextual understanding
        """
        response_parts = []
        
        # Add Prajna's interpretation
        if order.task_type == TonkaTaskType.CODE_GENERATION:
            response_parts.append(f"I'll {order.description} for you:")
        elif order.task_type == TonkaTaskType.PROJECT_CREATION:
            response_parts.append(f"I've created a {order.description} project structure:")
        elif order.task_type == TonkaTaskType.ALGORITHM_IMPLEMENTATION:
            response_parts.append(f"Here's the {order.description} implementation:")
        
        # Add the code
        if tonka_response.get("success"):
            code = tonka_response.get("code", "")
            if code:
                response_parts.append(f"\n\n```{order.language}\n{code}\n```")
            
            # Add explanation if available
            if prajna_context:
                response_parts.append(f"\n\n{prajna_context}")
            
            # Add confidence/source info
            confidence = tonka_response.get("confidence", 0)
            source = tonka_response.get("source", "generated")
            
            if confidence > 0.8:
                response_parts.append(f"\n\n✅ High confidence solution from {source}")
            elif confidence > 0.5:
                response_parts.append(f"\n\n✓ Good solution from {source}")
            else:
                response_parts.append(f"\n\n⚠️ Basic solution - you may want to enhance this")
        
        else:
            response_parts.append("\n\n❌ I encountered an issue generating the code. Let me try a different approach...")
        
        return "\n".join(response_parts)


# Singleton instance
coordinator = PrajnaTonkaCoordinator()


def analyze_and_route(user_query: str) -> Dict[str, Any]:
    """
    Main entry point for Prajna to analyze and route requests
    """
    # Extract all TONKA orders from the query
    orders = coordinator.extract_multiple_orders(user_query)
    
    if orders:
        logger.info(f"📋 Found {len(orders)} TONKA orders in query")
        return {
            "has_code_request": True,
            "orders": [order.to_dict() for order in orders],
            "primary_order": orders[0].to_dict() if orders else None,
            "requires_tonka": True
        }
    else:
        return {
            "has_code_request": False,
            "orders": [],
            "primary_order": None,
            "requires_tonka": False
        }


def create_tonka_request_from_query(user_query: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to create a TONKA request directly from a query
    """
    order = coordinator.analyze_request(user_query)
    if order:
        return coordinator.create_tonka_request(order)
    return None
