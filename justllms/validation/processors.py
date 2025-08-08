"""Rule processors for different types of content matching."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from justllms.validation.models import BusinessRule, RuleViolation


class BaseProcessor(ABC):
    """Base class for rule processors."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abstractmethod
    async def matches(self, content: str, rule: BusinessRule, context: Dict[str, Any]) -> bool:
        """Check if content matches the rule."""
        pass

    @abstractmethod
    async def find_matches(
        self, content: str, rule: BusinessRule, context: Dict[str, Any]
    ) -> List[Tuple[str, int, int]]:
        """Find all matches in content, returning (matched_text, start_pos, end_pos)."""
        pass

    async def create_violation(
        self, content: str, rule: BusinessRule, matches: List[Tuple[str, int, int]]
    ) -> Optional[RuleViolation]:
        """Create a violation object from matches."""
        if not matches:
            return None

        # Take the first match for the violation details
        matched_text, start_pos, end_pos = matches[0]

        return RuleViolation(
            rule_name=rule.name,
            rule_type=rule.type,
            action=rule.action,
            severity=rule.severity,
            message=rule.message,
            matched_content=matched_text,
            match_location={"start": start_pos, "end": end_pos},
            confidence=1.0,  # Most processors have perfect confidence
            metadata={
                "processor": self.name,
                "total_matches": len(matches),
                "all_matches": [
                    {"text": text, "start": start, "end": end} for text, start, end in matches
                ],
            },
        )


class KeywordProcessor(BaseProcessor):
    """Processor for keyword-based rules."""

    async def matches(self, content: str, rule: BusinessRule, context: Dict[str, Any]) -> bool:
        """Check if content contains any of the keywords."""
        if not rule.keywords:
            return False

        search_content = content if rule.case_sensitive else content.lower()
        search_keywords = (
            rule.keywords if rule.case_sensitive else [k.lower() for k in rule.keywords]
        )

        return any(keyword in search_content for keyword in search_keywords)

    async def find_matches(
        self, content: str, rule: BusinessRule, context: Dict[str, Any]
    ) -> List[Tuple[str, int, int]]:
        """Find all keyword matches in content."""
        if not rule.keywords:
            return []

        matches = []
        search_content = content if rule.case_sensitive else content.lower()
        search_keywords = (
            rule.keywords if rule.case_sensitive else [k.lower() for k in rule.keywords]
        )

        for keyword in search_keywords:
            start = 0
            while True:
                pos = search_content.find(keyword, start)
                if pos == -1:
                    break

                # Get the original text from content (preserving case)
                original_text = content[pos : pos + len(keyword)]
                matches.append((original_text, pos, pos + len(keyword)))
                start = pos + 1

        # Sort by position
        matches.sort(key=lambda x: x[1])
        return matches


class PatternMatcher(BaseProcessor):
    """Processor for regex pattern-based rules."""

    async def matches(self, content: str, rule: BusinessRule, context: Dict[str, Any]) -> bool:
        """Check if content matches any of the patterns."""
        if not rule.patterns:
            return False

        flags = 0 if rule.case_sensitive else re.IGNORECASE

        for pattern in rule.patterns:
            try:
                if re.search(pattern, content, flags):
                    return True
            except re.error:
                # Invalid regex pattern, skip
                continue

        return False

    async def find_matches(
        self, content: str, rule: BusinessRule, context: Dict[str, Any]
    ) -> List[Tuple[str, int, int]]:
        """Find all pattern matches in content."""
        if not rule.patterns:
            return []

        matches = []
        flags = 0 if rule.case_sensitive else re.IGNORECASE

        for pattern in rule.patterns:
            try:
                for match in re.finditer(pattern, content, flags):
                    matches.append((match.group(), match.start(), match.end()))
            except re.error:
                # Invalid regex pattern, skip
                continue

        # Sort by position
        matches.sort(key=lambda x: x[1])
        return matches


class ExactMatcher(BaseProcessor):
    """Processor for exact term matching."""

    async def matches(self, content: str, rule: BusinessRule, context: Dict[str, Any]) -> bool:
        """Check if content contains any exact terms."""
        if not rule.terms:
            return False

        # Split content into words for exact matching
        words = self._extract_words(content, rule.case_sensitive)
        search_terms = rule.terms if rule.case_sensitive else [t.lower() for t in rule.terms]

        return any(term in words for term in search_terms)

    async def find_matches(
        self, content: str, rule: BusinessRule, context: Dict[str, Any]
    ) -> List[Tuple[str, int, int]]:
        """Find all exact term matches in content."""
        if not rule.terms:
            return []

        matches = []
        search_terms = rule.terms if rule.case_sensitive else [t.lower() for t in rule.terms]

        # Use word boundaries for exact matching
        for term in search_terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            flags = 0 if rule.case_sensitive else re.IGNORECASE

            for match in re.finditer(pattern, content, flags):
                matches.append((match.group(), match.start(), match.end()))

        # Sort by position
        matches.sort(key=lambda x: x[1])
        return matches

    def _extract_words(self, content: str, case_sensitive: bool) -> set:
        """Extract individual words from content."""
        # Simple word extraction (could be enhanced)
        words = re.findall(r"\b\w+\b", content)
        if not case_sensitive:
            words = [w.lower() for w in words]
        return set(words)


class TopicClassifier(BaseProcessor):
    """Processor for topic-based classification rules."""

    def __init__(self) -> None:
        super().__init__()
        # Simple topic classification using keyword mapping
        # In production, this could use ML models
        self.topic_keywords = {
            "politics": [
                "election",
                "political",
                "politics",
                "government",
                "policy",
                "parliament",
                "congress",
            ],
            "sports": [
                "football",
                "soccer",
                "basketball",
                "cricket",
                "tennis",
                "sports",
                "game",
                "match",
                "tournament",
            ],
            "medical": [
                "doctor",
                "medicine",
                "health",
                "hospital",
                "disease",
                "treatment",
                "diagnosis",
            ],
            "financial": ["money", "investment", "stock", "finance", "banking", "loan", "credit"],
            "technology": [
                "computer",
                "software",
                "ai",
                "tech",
                "digital",
                "internet",
                "programming",
            ],
            "foreign_policy": [
                "diplomacy",
                "foreign policy",
                "international relations",
                "embassy",
                "trade deal",
            ],
            "legal": ["law", "legal", "court", "judge", "lawsuit", "attorney", "rights"],
            "entertainment": [
                "movie",
                "music",
                "celebrity",
                "entertainment",
                "film",
                "show",
                "actor",
            ],
        }

    async def matches(self, content: str, rule: BusinessRule, context: Dict[str, Any]) -> bool:
        """Check if content matches any of the specified topics."""
        if not rule.topics:
            return False

        detected_topics = await self._classify_topics(content, rule.case_sensitive)
        return any(topic in detected_topics for topic in rule.topics)

    async def find_matches(
        self, content: str, rule: BusinessRule, context: Dict[str, Any]
    ) -> List[Tuple[str, int, int]]:
        """Find topic-related content in text."""
        if not rule.topics:
            return []

        matches = []
        detected_topics = await self._classify_topics(content, rule.case_sensitive)

        # For each detected topic that matches our rule
        for topic in rule.topics:
            if topic in detected_topics:
                # Find the keywords that triggered this topic
                topic_keywords = self.topic_keywords.get(topic, [])
                for keyword in topic_keywords:
                    search_content = content if rule.case_sensitive else content.lower()
                    search_keyword = keyword if rule.case_sensitive else keyword.lower()

                    start = 0
                    while True:
                        pos = search_content.find(search_keyword, start)
                        if pos == -1:
                            break

                        # Get original text
                        original_text = content[pos : pos + len(keyword)]
                        matches.append(
                            (f"{original_text} (topic: {topic})", pos, pos + len(keyword))
                        )
                        start = pos + 1

        # Sort by position and remove duplicates
        unique_matches = []
        seen_positions = set()
        for match in sorted(matches, key=lambda x: x[1]):
            if match[1] not in seen_positions:
                unique_matches.append(match)
                seen_positions.add(match[1])

        return unique_matches

    async def _classify_topics(self, content: str, case_sensitive: bool) -> List[str]:
        """Simple topic classification based on keyword matching."""
        detected_topics = []
        search_content = content if case_sensitive else content.lower()

        for topic, keywords in self.topic_keywords.items():
            search_keywords = keywords if case_sensitive else [k.lower() for k in keywords]

            # Count keyword matches for this topic
            match_count = sum(1 for keyword in search_keywords if keyword in search_content)

            # If we find multiple keywords or content is short with any match
            if match_count >= 2 or (len(content.split()) < 20 and match_count >= 1):
                detected_topics.append(topic)

        return detected_topics


class IntentClassifier(BaseProcessor):
    """Processor for intent-based rules."""

    def __init__(self) -> None:
        super().__init__()
        # Simple intent patterns - could be enhanced with ML models
        self.intent_patterns = {
            "financial_advice": [
                r"(should|need to|recommend|suggest).*(buy|sell|invest)",
                r"(buy|sell|invest).*stock",
                r"financial advice",
                r"investment (tip|recommendation)",
            ],
            "medical_advice": [
                r"(should|need to|recommend|suggest).*(take|medicine|drug)",
                r"medical advice",
                r"(diagnose|treatment) for",
                r"what (medicine|drug) for",
            ],
            "legal_advice": [
                r"legal advice",
                r"(should|need to).*(sue|lawsuit|court)",
                r"what are my.*rights",
                r"(lawyer|attorney) recommend",
            ],
            "personal_information": [
                r"(my|your).*(address|phone|email|ssn)",
                r"personal information",
                r"private.*details",
            ],
        }

    async def matches(self, content: str, rule: BusinessRule, context: Dict[str, Any]) -> bool:
        """Check if content matches any specified intents."""
        if not rule.intents:
            return False

        detected_intents = await self._classify_intent(content, rule.case_sensitive)
        return any(intent in detected_intents for intent in rule.intents)

    async def find_matches(
        self, content: str, rule: BusinessRule, context: Dict[str, Any]
    ) -> List[Tuple[str, int, int]]:
        """Find intent-related content in text."""
        if not rule.intents:
            return []

        matches = []
        flags = 0 if rule.case_sensitive else re.IGNORECASE

        for intent in rule.intents:
            patterns = self.intent_patterns.get(intent, [])
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, content, flags):
                        matches.append(
                            (f"{match.group()} (intent: {intent})", match.start(), match.end())
                        )
                except re.error:
                    continue

        # Sort by position
        matches.sort(key=lambda x: x[1])
        return matches

    async def _classify_intent(self, content: str, case_sensitive: bool) -> List[str]:
        """Classify the intent of the content."""
        detected_intents = []
        flags = 0 if case_sensitive else re.IGNORECASE

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, content, flags):
                        detected_intents.append(intent)
                        break  # One match per intent is enough
                except re.error:
                    continue

        return detected_intents
