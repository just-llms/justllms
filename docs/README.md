# JustLLMs Documentation

This directory contains the Sphinx documentation for JustLLMs.

## Building Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Building HTML Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`. Open `index.html` in your browser to view it.

### Building Other Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# Plain text
make text

# EPUB
make epub
```

### Auto-rebuild During Development

For development, you can use sphinx-autobuild to automatically rebuild and refresh:

```bash
pip install sphinx-autobuild
sphinx-autobuild source build/html
```

Then open http://localhost:8000 in your browser.

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py          # Sphinx configuration
│   ├── index.rst        # Main documentation index
│   ├── quickstart.rst   # Quick start guide
│   ├── api/             # API reference (auto-generated)
│   │   ├── client.rst
│   │   ├── exceptions.rst
│   │   └── ...
│   └── ...
├── build/               # Generated documentation
└── Makefile            # Build commands
```

## Writing Documentation

### Adding New Pages

1. Create a new `.rst` file in the appropriate directory
2. Add it to the relevant `toctree` directive in `index.rst` or parent document
3. Follow the existing style and formatting

### API Documentation

API documentation is automatically generated from docstrings using Sphinx autodoc. Make sure to:

1. Write comprehensive docstrings for all public APIs
2. Use Google or NumPy style docstrings (both are supported)
3. Include examples in docstrings where appropriate

Example:

```python
def create_completion(self, messages: List[Message], model: str) -> CompletionResponse:
    """Create a completion using the specified model.
    
    Args:
        messages: List of messages in the conversation
        model: Model identifier (e.g., "gpt-3.5-turbo")
        
    Returns:
        CompletionResponse: The model's response
        
    Raises:
        ValidationError: If messages are invalid
        ProviderError: If the provider encounters an error
        
    Example:
        >>> client = Client()
        >>> response = client.create_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     model="gpt-3.5-turbo"
        ... )
        >>> print(response.content)
    """
```

### Documentation Style Guide

- Use clear, concise language
- Include code examples for all major features
- Explain both the "what" and the "why"
- Keep examples realistic and practical
- Update documentation when changing code

## Deploying Documentation

The documentation can be deployed to:

1. **Read the Docs**: Connect your GitHub repository to readthedocs.org
2. **GitHub Pages**: Use the `gh-pages` branch or `docs/` folder
3. **Custom hosting**: Upload the `build/html/` directory to any web server

## Contributing

When contributing to JustLLMs, please:

1. Update relevant documentation for any API changes
2. Add docstrings to new functions/classes
3. Include examples for new features
4. Run `make html` to ensure documentation builds without errors