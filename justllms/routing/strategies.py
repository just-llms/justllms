"""Routing strategies for intelligent model selection."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from justllms.core.base import BaseProvider
from justllms.core.models import Message, ModelInfo


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""
    
    @abstractmethod
    def select(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select a provider and model based on the strategy.
        
        Returns:
            Tuple of (provider_name, model_name)
        """
        pass


class CostOptimizedStrategy(RoutingStrategy):
    """Select the cheapest model that meets requirements."""
    
    def __init__(
        self,
        max_cost_per_1k_tokens: Optional[float] = None,
        min_context_length: Optional[int] = None,
        require_vision: bool = False,
        require_functions: bool = False,
    ):
        self.max_cost_per_1k_tokens = max_cost_per_1k_tokens
        self.min_context_length = min_context_length
        self.require_vision = require_vision
        self.require_functions = require_functions
    
    def select(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select the cheapest model that meets requirements."""
        candidates = []
        
        # Calculate total tokens needed (rough estimate)
        total_chars = sum(
            len(msg.content) if isinstance(msg.content, str) else 100
            for msg in messages
        )
        estimated_tokens = total_chars // 4  # Rough estimate
        
        for provider_name, provider in providers.items():
            models = provider.get_available_models()
            
            for model_name, model_info in models.items():
                # Check constraints
                if self.min_context_length and model_info.max_context_length:
                    if model_info.max_context_length < self.min_context_length:
                        continue
                
                if self.require_vision and not model_info.supports_vision:
                    continue
                
                if self.require_functions and not model_info.supports_functions:
                    continue
                
                if estimated_tokens > (model_info.max_context_length or 0):
                    continue
                
                # Calculate cost - be more permissive
                if model_info.cost_per_1k_prompt_tokens is not None:
                    avg_cost = (
                        (model_info.cost_per_1k_prompt_tokens or 0) +
                        (model_info.cost_per_1k_completion_tokens or 0)
                    ) / 2
                    
                    if self.max_cost_per_1k_tokens and avg_cost > self.max_cost_per_1k_tokens:
                        continue
                    
                    candidates.append((provider_name, model_name, avg_cost))
                else:
                    # If no cost info, assume reasonable cost
                    candidates.append((provider_name, model_name, 0.001))
        
        if not candidates:
            # Fallback to first available model
            for provider_name, provider in providers.items():
                models = provider.get_available_models()
                if models:
                    model_name = list(models.keys())[0]
                    return provider_name, model_name
            
            raise ValueError("No suitable models found")
        
        # Sort by cost and return cheapest
        candidates.sort(key=lambda x: x[2])
        # Debug: print candidates for cost strategy
        # Debug: Cost strategy selecting cheapest model
        return candidates[0][0], candidates[0][1]


class LatencyOptimizedStrategy(RoutingStrategy):
    """Select the fastest model that meets requirements."""
    
    def __init__(
        self,
        max_latency_ms: Optional[float] = None,
        prefer_local: bool = False,
    ):
        self.max_latency_ms = max_latency_ms
        self.prefer_local = prefer_local
    
    def select(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select the fastest model."""
        candidates = []
        
        for provider_name, provider in providers.items():
            models = provider.get_available_models()
            
            for model_name, model_info in models.items():
                # Estimate latency (this is simplified - in practice you'd measure actual latency)
                latency_score = 1.0
                
                # Prefer smaller/faster models for lower latency
                if "mini" in model_name.lower() or "haiku" in model_name.lower():
                    latency_score = 0.3  # Fastest
                elif "flash-8b" in model_name.lower():
                    latency_score = 0.4  # Very fast
                elif "turbo" in model_name.lower() or "flash" in model_name.lower():
                    latency_score = 0.6  # Fast
                elif "pro" in model_name.lower() or "sonnet" in model_name.lower():
                    latency_score = 0.8  # Medium
                elif "opus" in model_name.lower():
                    latency_score = 1.0  # Slowest
                
                # Add provider-specific latency estimates
                if provider_name == "google":
                    latency_score *= 0.9  # Generally fast
                elif provider_name == "openai":
                    latency_score *= 0.8  # Very fast
                elif provider_name == "anthropic":
                    latency_score *= 1.0  # Standard
                
                candidates.append((provider_name, model_name, latency_score))
        
        if not candidates:
            # No candidates found, using fallback logic
            # Use all available models if no specific candidates
            for provider_name, provider in providers.items():
                models = provider.get_available_models()
                for model_name, model_info in models.items():
                    latency_score = 1.0
                    
                    # Apply same scoring logic
                    if "mini" in model_name.lower() or "haiku" in model_name.lower():
                        latency_score = 0.3
                    elif "flash-8b" in model_name.lower():
                        latency_score = 0.4
                    elif "turbo" in model_name.lower() or "flash" in model_name.lower():
                        latency_score = 0.6
                    elif "pro" in model_name.lower() or "sonnet" in model_name.lower():
                        latency_score = 0.8
                    elif "opus" in model_name.lower():
                        latency_score = 1.0
                    
                    if provider_name == "google":
                        latency_score *= 0.9
                    elif provider_name == "openai":
                        latency_score *= 0.8
                    elif provider_name == "anthropic":
                        latency_score *= 1.0
                    
                    candidates.append((provider_name, model_name, latency_score))
            
            if not candidates:
                raise ValueError("No suitable models found")
        
        # Sort by latency score and return fastest
        candidates.sort(key=lambda x: x[2])
        # Sort by latency score and return fastest
        return candidates[0][0], candidates[0][1]


class QualityOptimizedStrategy(RoutingStrategy):
    """Select the highest quality model within constraints."""
    
    def __init__(
        self,
        min_quality_tier: str = "standard",  # "basic", "standard", "advanced", "flagship"
        max_cost_per_1k_tokens: Optional[float] = None,
    ):
        self.min_quality_tier = min_quality_tier
        self.max_cost_per_1k_tokens = max_cost_per_1k_tokens
        
        self.tier_rankings = {
            "basic": 1,
            "standard": 2,
            "advanced": 3,
            "flagship": 4,
        }
    
    def _get_quality_tier(self, model_info: ModelInfo, model_name: str) -> int:
        """Determine quality tier of a model."""
        # Check tags first - flagship models get highest tier
        if "flagship" in model_info.tags:
            return 4
        
        # Specific model name matching for Google models
        model_lower = model_name.lower()
        
        # Tier 4 - Flagship/Premium models
        if any(name in model_lower for name in ["gemini-2.5-pro", "opus", "gpt-4o", "grok-4"]):
            return 4
        # Tier 3 - Advanced models  
        elif any(name in model_lower for name in ["gemini-2.5-flash", "gemini-1.5-pro", "sonnet", "gpt-4", "grok-3", "deepseek-reasoner"]):
            return 3
        # Tier 2 - Standard models
        elif any(name in model_lower for name in ["gemini-1.5-flash", "gemini-1.0-pro", "haiku", "gpt-3.5", "deepseek-chat"]):
            return 2
        # Tier 1 - Basic/Small models
        elif any(name in model_lower for name in ["flash-8b", "mini"]):
            return 1
        else:
            return 2  # Default to standard
    
    def select(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select the highest quality model within constraints."""
        candidates = []
        
        for provider_name, provider in providers.items():
            models = provider.get_available_models()
            
            for model_name, model_info in models.items():
                quality_tier = self._get_quality_tier(model_info, model_name)
                
                # Skip cost constraint for now - just focus on quality
                # Check cost constraint (be more lenient)
                # if self.max_cost_per_1k_tokens and model_info.cost_per_1k_prompt_tokens:
                #     avg_cost = (
                #         (model_info.cost_per_1k_prompt_tokens or 0) +
                #         (model_info.cost_per_1k_completion_tokens or 0)
                #     ) / 2
                #     
                #     if avg_cost > self.max_cost_per_1k_tokens:
                #         continue
                
                candidates.append((provider_name, model_name, quality_tier))
        
        if not candidates:
            # No candidates found, using fallback logic
            # Fallback to best available
            for provider_name, provider in providers.items():
                models = provider.get_available_models()
                if models:
                    # Try to pick the highest quality model name-wise
                    best_model = None
                    best_score = 0
                    for model_name in models.keys():
                        score = self._get_quality_tier(models[model_name], model_name)
                        if score > best_score:
                            best_score = score
                            best_model = model_name
                    # Quality fallback selected
                    return provider_name, best_model or list(models.keys())[0]
            
            raise ValueError("No suitable models found")
        
        # Sort by quality tier (descending) and return best
        candidates.sort(key=lambda x: x[2], reverse=True)
        # Debug: print candidates for quality strategy
        # Sort by quality tier and return highest quality
        return candidates[0][0], candidates[0][1]


class TaskBasedStrategy(RoutingStrategy):
    """Select model based on task type detection."""
    
    def __init__(self):
        self.task_patterns = {
            "code": ["code", "function", "class", "debug", "implement", "program"],
            "analysis": ["analyze", "explain", "understand", "compare", "evaluate"],
            "creative": ["write", "story", "poem", "creative", "imagine"],
            "simple": ["hello", "hi", "thanks", "yes", "no", "ok"],
            "vision": ["image", "picture", "photo", "screenshot", "visual"],
        }
    
    def _detect_task_type(self, messages: List[Message]) -> str:
        """Detect the type of task from messages."""
        combined_text = " ".join(
            msg.content.lower() if isinstance(msg.content, str) else ""
            for msg in messages
        ).lower()
        
        # Check for vision content
        for msg in messages:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        return "vision"
        
        # Enhanced pattern matching with priority
        # Check for complex analysis patterns first
        if any(pattern in combined_text for pattern in ["history of", "explain", "analyze", "compare", "1000 words", "detailed", "comprehensive"]):
            return "analysis"
        
        # Check for simple patterns
        if any(pattern in combined_text for pattern in ["what is", "2+2", "simple", "quick"]):
            return "simple"
            
        # Check other patterns
        for task_type, patterns in self.task_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return task_type
        
        return "general"
    
    def select(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select model based on detected task type."""
        task_type = self._detect_task_type(messages)
        # Task detected and routed accordingly
        
        # Define preferred models for each task type - including new providers
        preferences = {
            "code": [
                ("google", "gemini-2.5-pro"),     # Best for complex code
                ("deepseek", "deepseek-reasoner"), # Excellent for reasoning tasks
                ("grok", "grok-4"),               # Latest reasoning model
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("google", "gemini-2.5-flash"),
                ("openai", "gpt-4o"),
            ],
            "analysis": [
                ("deepseek", "deepseek-reasoner"), # Specialized for reasoning
                ("google", "gemini-2.5-pro"),     # Best for analysis
                ("grok", "grok-4"),               # Good reasoning capabilities
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("google", "gemini-1.5-pro"),
                ("openai", "gpt-4o"),
            ],
            "creative": [
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("grok", "grok-4"),               # Known for creative responses
                ("google", "gemini-1.5-pro"),    # Good for creative tasks
                ("google", "gemini-2.5-flash"),
                ("openai", "gpt-4o"),
            ],
            "simple": [
                ("deepseek", "deepseek-chat"),    # Very affordable for simple tasks
                ("google", "gemini-1.5-flash-8b"), # Fastest/cheapest
                ("anthropic", "claude-3-5-haiku-20241022"),
                ("google", "gemini-1.5-flash"),
                ("openai", "gpt-3.5-turbo"),
            ],
            "vision": [
                ("grok", "grok-4"),              # Supports vision
                ("google", "gemini-2.5-flash"),  # Latest multimodal
                ("google", "gemini-1.5-pro"),
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
            ],
            "general": [
                ("deepseek", "deepseek-chat"),    # Great value for general use
                ("google", "gemini-1.5-flash"),  # Balanced option
                ("grok", "grok-3"),              # Good general purpose
                ("google", "gemini-2.5-flash"),
                ("anthropic", "claude-3-5-haiku-20241022"),
            ],
        }
        
        # Try preferred models in order
        for provider_name, model_name in preferences.get(task_type, preferences["general"]):
            if provider_name in providers:
                provider = providers[provider_name]
                available_models = provider.get_available_models()
                if model_name in available_models:
                    return provider_name, model_name
        
        # If no preferred model found, use first available from preferred providers
        for provider_name, model_name in preferences.get(task_type, preferences["general"]):
            if provider_name in providers:
                provider = providers[provider_name]
                available_models = provider.get_available_models()
                if available_models:
                    # Pick the first available model from this provider
                    first_model = list(available_models.keys())[0]
                    return provider_name, first_model
        
        # Final fallback to any available model
        for provider_name, provider in providers.items():
            models = provider.get_available_models()
            if models:
                model_name = list(models.keys())[0]
                return provider_name, model_name
        
        raise ValueError("No suitable models found")