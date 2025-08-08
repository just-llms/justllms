"""Validation data models and configuration."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationAction(str, Enum):
    """Actions to take when a rule is violated."""

    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"
    REPLACE = "replace"
    FLAG = "flag"


class RuleType(str, Enum):
    """Types of business rules supported."""

    KEYWORDS = "keywords"
    PATTERNS = "patterns"
    EXACT_MATCHES = "exact_matches"
    TOPICS = "topics"
    INTENT = "intent"


class RuleSeverity(str, Enum):
    """Severity levels for rule violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BusinessRule(BaseModel):
    """Configuration for a single business rule."""

    name: str = Field(..., description="Unique rule identifier")
    description: Optional[str] = Field(default=None, description="Human-readable rule description")
    type: RuleType = Field(..., description="Type of rule processor to use")

    # Rule content (different fields used based on type)
    keywords: Optional[List[str]] = Field(
        default=None, description="Keywords to match (for keyword rules)"
    )
    patterns: Optional[List[str]] = Field(
        default=None, description="Regex patterns to match (for pattern rules)"
    )
    terms: Optional[List[str]] = Field(
        default=None, description="Exact terms to match (for exact_matches rules)"
    )
    topics: Optional[List[str]] = Field(
        default=None, description="Topic categories to match (for topic rules)"
    )
    intents: Optional[List[str]] = Field(
        default=None, description="Intent categories to match (for intent rules)"
    )

    # Rule behavior
    action: ValidationAction = Field(
        default=ValidationAction.BLOCK, description="Action to take on violation"
    )
    message: Optional[str] = Field(
        default=None, description="Message to show/log when rule is violated"
    )
    replacement: Optional[str] = Field(
        default=None, description="Replacement text (for replace action)"
    )
    severity: RuleSeverity = Field(
        default=RuleSeverity.MEDIUM, description="Severity level of violations"
    )

    # Rule control
    enabled: bool = Field(default=True, description="Whether rule is active")
    case_sensitive: bool = Field(
        default=False, description="Whether matching should be case sensitive"
    )

    # Context filters (when to apply this rule)
    apply_to_providers: Optional[List[str]] = Field(
        default=None, description="Only apply to specific providers"
    )
    apply_to_models: Optional[List[str]] = Field(
        default=None, description="Only apply to specific models"
    )
    apply_to_users: Optional[List[str]] = Field(
        default=None, description="Only apply to specific users"
    )

    # Metadata
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now, description="Rule creation timestamp"
    )
    created_by: Optional[str] = Field(default=None, description="Who created this rule")
    tags: List[str] = Field(default_factory=list, description="Tags for rule organization")

    def applies_to_context(self, context: Dict[str, Any]) -> bool:
        """Check if this rule should be applied in the given context."""
        # Provider filter
        if self.apply_to_providers and context.get("provider") not in self.apply_to_providers:
            return False

        # Model filter
        if self.apply_to_models and context.get("model") not in self.apply_to_models:
            return False

        # User filter
        return not (self.apply_to_users and context.get("user_id") not in self.apply_to_users)

    def get_rule_content(self) -> List[str]:
        """Get the content to match based on rule type."""
        if self.type == RuleType.KEYWORDS:
            return self.keywords or []
        elif self.type == RuleType.PATTERNS:
            return self.patterns or []
        elif self.type == RuleType.EXACT_MATCHES:
            return self.terms or []
        elif self.type == RuleType.TOPICS:
            return self.topics or []
        elif self.type == RuleType.INTENT:
            return self.intents or []
        else:
            return []


class RuleViolation(BaseModel):
    """Represents a rule violation."""

    rule_name: str = Field(..., description="Name of the violated rule")
    rule_type: RuleType = Field(..., description="Type of rule that was violated")
    action: ValidationAction = Field(..., description="Action taken for this violation")
    severity: RuleSeverity = Field(..., description="Severity of the violation")
    message: Optional[str] = Field(default=None, description="Violation message")
    matched_content: Optional[str] = Field(
        default=None, description="Content that triggered the rule"
    )
    match_location: Optional[Dict[str, int]] = Field(
        default=None, description="Where in content the match occurred"
    )
    confidence: Optional[float] = Field(default=None, description="Confidence score for the match")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional violation metadata"
    )


class ValidationResult(BaseModel):
    """Result of validation pipeline execution."""

    passed: bool = Field(..., description="Whether validation passed overall")
    violations: List[RuleViolation] = Field(
        default_factory=list, description="List of rule violations found"
    )
    processed_at: datetime = Field(
        default_factory=datetime.now, description="When validation was performed"
    )
    processing_time_ms: Optional[float] = Field(default=None, description="Time taken to process")

    # Actions taken
    blocked: bool = Field(default=False, description="Whether response was blocked")
    modified: bool = Field(default=False, description="Whether response was modified")
    flagged: bool = Field(default=False, description="Whether response was flagged for review")

    # Modified content
    original_content: Optional[str] = Field(default=None, description="Original response content")
    modified_content: Optional[str] = Field(default=None, description="Modified response content")

    # Metadata
    context: Dict[str, Any] = Field(default_factory=dict, description="Validation context")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional validation metadata"
    )

    @property
    def should_block(self) -> bool:
        """Check if response should be blocked based on violations."""
        return any(v.action == ValidationAction.BLOCK for v in self.violations)

    @property
    def should_modify(self) -> bool:
        """Check if response should be modified."""
        return any(
            v.action in [ValidationAction.REDACT, ValidationAction.REPLACE] for v in self.violations
        )

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(v.action == ValidationAction.WARN for v in self.violations)

    @property
    def has_flags(self) -> bool:
        """Check if there are any flags for review."""
        return any(v.action == ValidationAction.FLAG for v in self.violations)

    @property
    def highest_severity(self) -> Optional[RuleSeverity]:
        """Get the highest severity level among violations."""
        if not self.violations:
            return None

        severity_order = {
            RuleSeverity.LOW: 1,
            RuleSeverity.MEDIUM: 2,
            RuleSeverity.HIGH: 3,
            RuleSeverity.CRITICAL: 4,
        }

        return max(self.violations, key=lambda v: severity_order[v.severity]).severity

    def get_violations_by_action(self, action: ValidationAction) -> List[RuleViolation]:
        """Get violations that require a specific action."""
        return [v for v in self.violations if v.action == action]

    def get_violations_by_severity(self, min_severity: RuleSeverity) -> List[RuleViolation]:
        """Get violations above a certain severity level."""
        severity_order = {
            RuleSeverity.LOW: 1,
            RuleSeverity.MEDIUM: 2,
            RuleSeverity.HIGH: 3,
            RuleSeverity.CRITICAL: 4,
        }

        min_level = severity_order[min_severity]
        return [v for v in self.violations if severity_order[v.severity] >= min_level]


class ValidationConfig(BaseModel):
    """Configuration for the validation system."""

    enabled: bool = Field(default=True, description="Whether validation is enabled")
    business_rules: List[BusinessRule] = Field(
        default_factory=list, description="List of business rules to apply"
    )

    # Processing settings
    max_processing_time_ms: int = Field(default=5000, description="Max time to spend on validation")
    concurrent_processing: bool = Field(
        default=True, description="Whether to process rules concurrently"
    )

    # Logging and monitoring
    log_violations: bool = Field(default=True, description="Whether to log rule violations")
    log_passed_validations: bool = Field(
        default=False, description="Whether to log successful validations"
    )
    store_validation_history: bool = Field(
        default=True, description="Whether to store validation history"
    )

    # Error handling
    fail_fast: bool = Field(default=False, description="Whether to stop on first violation")
    error_on_processor_failure: bool = Field(
        default=False, description="Whether to error if a processor fails"
    )

    # Default actions
    default_block_message: str = Field(
        default="I cannot provide that information due to content restrictions.",
        description="Default message for blocked responses",
    )
    default_warning_message: str = Field(
        default="Please note: This response may contain sensitive information.",
        description="Default message for warnings",
    )

    def get_enabled_rules(self) -> List[BusinessRule]:
        """Get only the enabled rules."""
        return [rule for rule in self.business_rules if rule.enabled]

    def get_rules_by_type(self, rule_type: RuleType) -> List[BusinessRule]:
        """Get rules of a specific type."""
        return [rule for rule in self.business_rules if rule.type == rule_type and rule.enabled]

    def get_rule_by_name(self, name: str) -> Optional[BusinessRule]:
        """Get a rule by its name."""
        for rule in self.business_rules:
            if rule.name == name:
                return rule
        return None

    def add_rule(self, rule: BusinessRule) -> None:
        """Add a new rule to the configuration."""
        # Remove existing rule with same name if it exists
        self.business_rules = [r for r in self.business_rules if r.name != rule.name]
        self.business_rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        original_count = len(self.business_rules)
        self.business_rules = [r for r in self.business_rules if r.name != name]
        return len(self.business_rules) < original_count

    def disable_rule(self, name: str) -> bool:
        """Disable a rule by name."""
        rule = self.get_rule_by_name(name)
        if rule:
            rule.enabled = False
            return True
        return False

    def enable_rule(self, name: str) -> bool:
        """Enable a rule by name."""
        rule = self.get_rule_by_name(name)
        if rule:
            rule.enabled = True
            return True
        return False
