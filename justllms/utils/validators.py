from typing import Any, Dict, List, Union

from justllms.core.models import Message, Role
from justllms.exceptions import ValidationError


def validate_messages(  # noqa: C901
    messages: Union[List[Dict[str, Any]], List[Message]],
) -> List[Message]:
    """Validate and normalize message inputs for LLM requests.

    Performs comprehensive validation of message structure, content, roles,
    and conversation flow to ensure compatibility with provider APIs.
    Converts dictionary inputs to Message objects.

    Args:
        messages: List of messages as dictionaries or Message objects.
                 Each message must have 'role' and 'content' fields.

    Returns:
        List[Message]: Validated and normalized Message objects ready for
                      provider consumption.

    Raises:
        ValidationError: If messages are invalid, empty, malformed, or missing
                        required fields. Includes specific error descriptions
                        and field references for debugging.
    """
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
