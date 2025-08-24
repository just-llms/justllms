"""Validation utilities for inputs."""

import re
from typing import Any, Dict, List, Optional, Union

from justllms.core.models import Message, Role
from justllms.exceptions import ValidationError


def validate_messages(  # noqa: C901
    messages: Union[List[Dict[str, Any]], List[Message]],
) -> List[Message]:
    """Validate and convert messages to Message objects."""
    if not messages:
        raise ValidationError("Messages list cannot be empty")

    if not isinstance(messages, list):
        raise ValidationError("Messages must be a list")

    validated_messages = []

    for i, msg in enumerate(messages):
        if isinstance(msg, Message):
            validated_messages.append(msg)
        elif isinstance(msg, dict):
            # Validate required fields
            if "role" not in msg:
                raise ValidationError(f"Message {i} missing required field 'role'")

            if "content" not in msg:
                raise ValidationError(f"Message {i} missing required field 'content'")

            # Validate role
            role = msg["role"]
            if isinstance(role, str):
                try:
                    role = Role(role.lower())
                except ValueError as e:
                    valid_roles = [r.value for r in Role]
                    raise ValidationError(
                        f"Message {i} has invalid role '{role}'. "
                        f"Valid roles are: {', '.join(valid_roles)}"
                    ) from e
            elif not isinstance(role, Role):
                raise ValidationError(f"Message {i} role must be a string or Role enum")

            # Validate content
            content = msg["content"]
            if not isinstance(content, (str, list)):
                raise ValidationError(f"Message {i} content must be a string or list")

            if isinstance(content, str) and not content.strip():
                raise ValidationError(f"Message {i} content cannot be empty")

            if isinstance(content, list):
                if not content:
                    raise ValidationError(f"Message {i} content list cannot be empty")

                # Validate multimodal content
                for j, item in enumerate(content):
                    if not isinstance(item, dict):
                        raise ValidationError(f"Message {i} content item {j} must be a dictionary")

                    if "type" not in item:
                        raise ValidationError(f"Message {i} content item {j} missing 'type' field")

                    item_type = item["type"]
                    if item_type == "text":
                        if "text" not in item:
                            raise ValidationError(
                                f"Message {i} content item {j} of type 'text' missing 'text' field"
                            )
                    elif item_type == "image":
                        if "image" not in item and "image_url" not in item:
                            raise ValidationError(
                                f"Message {i} content item {j} of type 'image' "
                                "missing 'image' or 'image_url' field"
                            )
                    else:
                        # Allow other types but don't validate
                        pass

            # Create Message object
            try:
                validated_messages.append(Message(**msg))
            except Exception as e:
                raise ValidationError(f"Message {i} validation failed: {str(e)}") from e
        else:
            raise ValidationError(f"Message {i} must be a dict or Message object, got {type(msg)}")

    # Additional validations
    if not any(msg.role == Role.USER for msg in validated_messages):
        raise ValidationError("Messages must contain at least one user message")

    # Check message order (system messages should be first)
    system_indices = [i for i, msg in enumerate(validated_messages) if msg.role == Role.SYSTEM]

    if system_indices and any(i > 0 for i in system_indices):
        # Allow system messages after position 0 but warn
        pass

    return validated_messages


def validate_model_name(model: str) -> str:
    """Validate and normalize model name."""
    if not model:
        raise ValidationError("Model name cannot be empty")

    if not isinstance(model, str):
        raise ValidationError(f"Model name must be a string, got {type(model)}")

    model = model.strip()

    # Check for common issues
    if len(model) > 100:
        raise ValidationError("Model name is too long (max 100 characters)")

    # Allow provider/model format
    if "/" in model:
        parts = model.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValidationError("Model name with provider must be in format 'provider/model'")

        provider, model_name = parts

        # Validate provider name
        if not re.match(r"^[a-zA-Z0-9_-]+$", provider):
            raise ValidationError(
                f"Invalid provider name '{provider}'. "
                "Must contain only letters, numbers, hyphens, and underscores"
            )

        # Validate model name
        if not re.match(r"^[a-zA-Z0-9_.-]+$", model_name):
            raise ValidationError(
                f"Invalid model name '{model_name}'. "
                "Must contain only letters, numbers, hyphens, periods, and underscores"
            )
    else:
        # Validate standalone model name
        if not re.match(r"^[a-zA-Z0-9_.-]+$", model):
            raise ValidationError(
                f"Invalid model name '{model}'. "
                "Must contain only letters, numbers, hyphens, periods, and underscores"
            )

    return model


def validate_temperature(temperature: float) -> float:
    """Validate temperature parameter."""
    if not isinstance(temperature, (int, float)):
        raise ValidationError(f"Temperature must be a number, got {type(temperature)}")

    if temperature < 0 or temperature > 2:
        raise ValidationError(f"Temperature must be between 0 and 2, got {temperature}")

    return float(temperature)


def validate_max_tokens(max_tokens: int) -> int:
    """Validate max_tokens parameter."""
    if not isinstance(max_tokens, int):
        raise ValidationError(f"max_tokens must be an integer, got {type(max_tokens)}")

    if max_tokens < 1:
        raise ValidationError(f"max_tokens must be positive, got {max_tokens}")

    if max_tokens > 1000000:  # Reasonable upper limit
        raise ValidationError(f"max_tokens is too large (max 1000000), got {max_tokens}")

    return max_tokens


def validate_stop_sequences(stop: Union[str, List[str], None]) -> Optional[List[str]]:
    """Validate stop sequences."""
    if stop is None:
        return None

    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        raise ValidationError(f"stop must be a string or list of strings, got {type(stop)}")

    validated_stop = []
    for i, seq in enumerate(stop):
        if not isinstance(seq, str):
            raise ValidationError(f"stop sequence {i} must be a string, got {type(seq)}")

        if not seq:
            raise ValidationError(f"stop sequence {i} cannot be empty")

        if len(seq) > 500:  # Reasonable limit
            raise ValidationError(f"stop sequence {i} is too long (max 500 characters)")

        validated_stop.append(seq)

    if len(validated_stop) > 20:  # Reasonable limit
        raise ValidationError("Too many stop sequences (max 20)")

    return validated_stop
