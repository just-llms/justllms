"""Business rule engine for response validation."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from justllms.validation.models import (
    BusinessRule,
    RuleType,
    RuleViolation,
    ValidationAction,
    ValidationConfig,
    ValidationResult,
)
from justllms.validation.processors import (
    BaseProcessor,
    ExactMatcher,
    IntentClassifier,
    KeywordProcessor,
    PatternMatcher,
    TopicClassifier,
)


class BusinessRuleEngine:
    """Generic business rule engine for response validation."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

        # Initialize processors
        self.processors: Dict[RuleType, BaseProcessor] = {
            RuleType.KEYWORDS: KeywordProcessor(),
            RuleType.PATTERNS: PatternMatcher(),
            RuleType.EXACT_MATCHES: ExactMatcher(),
            RuleType.TOPICS: TopicClassifier(),
            RuleType.INTENT: IntentClassifier(),
        }

        # Statistics
        self.validation_count = 0
        self.violation_count = 0
        self.total_processing_time = 0.0

    async def validate(  # noqa: C901
        self, content: str, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate content against all enabled business rules."""
        if not self.config.enabled:
            return ValidationResult(passed=True)

        context = context or {}
        start_time = time.time()
        violations = []

        try:
            # Get applicable rules
            applicable_rules = [
                rule for rule in self.config.get_enabled_rules() if rule.applies_to_context(context)
            ]

            if not applicable_rules:
                return ValidationResult(passed=True, context=context)

            # Process rules (concurrent or sequential)
            if self.config.concurrent_processing:
                violations = await self._process_rules_concurrent(
                    content, applicable_rules, context
                )
            else:
                violations = await self._process_rules_sequential(
                    content, applicable_rules, context
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Check if we exceeded max processing time
            if processing_time > self.config.max_processing_time_ms:
                # Log warning but don't fail validation
                pass

            # Create validation result
            result = ValidationResult(
                passed=len(violations) == 0,
                violations=violations,
                processing_time_ms=processing_time,
                blocked=any(v.action == ValidationAction.BLOCK for v in violations),
                modified=any(
                    v.action in [ValidationAction.REDACT, ValidationAction.REPLACE]
                    for v in violations
                ),
                flagged=any(v.action == ValidationAction.FLAG for v in violations),
                original_content=content,
                context=context,
            )

            # Apply content modifications if needed
            if result.should_modify:
                result.modified_content = await self._apply_content_modifications(
                    content, violations
                )
                result.modified = True

            # Update statistics
            self.validation_count += 1
            if violations:
                self.violation_count += 1
            self.total_processing_time += processing_time

            # Log if configured
            if self.config.log_violations and violations:
                self._log_violations(content, violations, context)
            elif self.config.log_passed_validations and not violations:
                self._log_passed_validation(content, context)

            return result

        except Exception as e:
            # Handle validation errors
            if self.config.error_on_processor_failure:
                raise e

            # Return failed validation with error info
            return ValidationResult(
                passed=False,
                violations=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                context=context,
                metadata={"error": str(e)},
            )

    async def _process_rules_concurrent(
        self, content: str, rules: List[BusinessRule], context: Dict[str, Any]
    ) -> List[RuleViolation]:
        """Process rules concurrently for better performance."""
        tasks = []

        for rule in rules:
            task = asyncio.create_task(self._check_single_rule(content, rule, context))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        violations = []
        for result in results:
            if isinstance(result, RuleViolation):
                violations.append(result)
            elif isinstance(result, Exception) and self.config.error_on_processor_failure:
                # Log error but continue processing
                raise result

        return violations

    async def _process_rules_sequential(
        self, content: str, rules: List[BusinessRule], context: Dict[str, Any]
    ) -> List[RuleViolation]:
        """Process rules sequentially."""
        violations = []

        for rule in rules:
            try:
                violation = await self._check_single_rule(content, rule, context)
                if violation:
                    violations.append(violation)

                    # Fast fail if configured
                    if self.config.fail_fast and violation.action == ValidationAction.BLOCK:
                        break

            except Exception as e:
                if self.config.error_on_processor_failure:
                    raise e
                # Continue with other rules

        return violations

    async def _check_single_rule(
        self, content: str, rule: BusinessRule, context: Dict[str, Any]
    ) -> Optional[RuleViolation]:
        """Check a single rule against content."""
        processor = self.processors.get(rule.type)
        if not processor:
            return None

        try:
            # Check if rule matches
            if await processor.matches(content, rule, context):
                # Find detailed matches
                matches = await processor.find_matches(content, rule, context)

                # Create violation
                violation = await processor.create_violation(content, rule, matches)
                return violation

        except Exception as e:
            # Log error but don't fail the entire validation
            if self.config.error_on_processor_failure:
                raise e

        return None

    async def _apply_content_modifications(
        self, content: str, violations: List[RuleViolation]
    ) -> str:
        """Apply content modifications based on violations."""
        modified_content = content

        # Process violations in reverse order of position to avoid offset issues
        redact_violations = [v for v in violations if v.action == ValidationAction.REDACT]
        replace_violations = [v for v in violations if v.action == ValidationAction.REPLACE]

        # Sort by position (reverse order)
        all_modifications = redact_violations + replace_violations
        all_modifications.sort(key=lambda v: (v.match_location or {}).get("start", 0), reverse=True)

        for violation in all_modifications:
            if not violation.match_location:
                continue

            start = violation.match_location["start"]
            end = violation.match_location["end"]

            if violation.action == ValidationAction.REDACT:
                # Replace with [REDACTED]
                replacement = "[REDACTED]"
            elif violation.action == ValidationAction.REPLACE:
                # Use custom replacement from rule
                rule = self.get_rule_by_name(violation.rule_name)
                replacement = rule.replacement if rule and rule.replacement else "[REPLACED]"
            else:
                continue

            # Apply modification
            modified_content = modified_content[:start] + replacement + modified_content[end:]

        return modified_content

    def get_rule_by_name(self, name: str) -> Optional[BusinessRule]:
        """Get a rule by its name."""
        return self.config.get_rule_by_name(name)

    def add_rule(self, rule: BusinessRule) -> None:
        """Add a new rule to the engine."""
        self.config.add_rule(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        return self.config.remove_rule(name)

    def enable_rule(self, name: str) -> bool:
        """Enable a rule by name."""
        return self.config.enable_rule(name)

    def disable_rule(self, name: str) -> bool:
        """Disable a rule by name."""
        return self.config.disable_rule(name)

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation engine statistics."""
        avg_processing_time = (
            self.total_processing_time / self.validation_count if self.validation_count > 0 else 0
        )

        violation_rate = (
            self.violation_count / self.validation_count if self.validation_count > 0 else 0
        ) * 100

        return {
            "total_validations": self.validation_count,
            "total_violations": self.violation_count,
            "violation_rate_percent": violation_rate,
            "average_processing_time_ms": avg_processing_time,
            "total_processing_time_ms": self.total_processing_time,
            "enabled_rules_count": len(self.config.get_enabled_rules()),
            "total_rules_count": len(self.config.business_rules),
        }

    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.validation_count = 0
        self.violation_count = 0
        self.total_processing_time = 0.0

    def _log_violations(
        self, content: str, violations: List[RuleViolation], context: Dict[str, Any]
    ) -> None:
        """Log rule violations."""
        # Simple logging - could be enhanced with proper logging framework
        # Log validation violations - could be enhanced with proper logging framework
        pass

    def _log_passed_validation(self, content: str, context: Dict[str, Any]) -> None:
        """Log successful validation."""
        # Content passed all validation rules
        pass

    async def test_rule(
        self, rule: BusinessRule, test_content: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[RuleViolation]:
        """Test a single rule against content (useful for rule development)."""
        context = context or {}

        if not rule.applies_to_context(context):
            return None

        return await self._check_single_rule(test_content, rule, context)

    async def explain_validation(
        self, content: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get detailed explanation of validation results."""
        result = await self.validate(content, context)

        explanation: Dict[str, Any] = {
            "content": content,
            "passed": result.passed,
            "total_rules_checked": len(self.config.get_enabled_rules()),
            "violations_found": len(result.violations),
            "processing_time_ms": result.processing_time_ms,
            "actions_taken": {
                "blocked": result.blocked,
                "modified": result.modified,
                "flagged": result.flagged,
            },
            "violation_details": [],
        }

        for violation in result.violations:
            explanation["violation_details"].append(
                {
                    "rule_name": violation.rule_name,
                    "rule_type": violation.rule_type.value,
                    "action": violation.action.value,
                    "severity": violation.severity.value,
                    "message": violation.message,
                    "matched_content": violation.matched_content,
                    "confidence": violation.confidence,
                }
            )

        if result.modified_content:
            explanation["modified_content"] = result.modified_content

        return explanation
