# scurl

[![PyPI version](https://badge.fury.io/py/scurl.svg)](https://badge.fury.io/py/scurl)
[![CI](https://github.com/yourusername/scurl/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/scurl/actions/workflows/ci.yml)

A secure curl wrapper with middleware support, HTML-to-markdown extraction, and prompt injection detection.

## Installation

```bash
pip install sibylline-scurl
```

Or with [pipx](https://pipx.pypa.io/) (recommended for CLI tools):

```bash
pipx install sibylline-scurl
```

For prompt injection detection, install the optional dependencies:

```bash
pip install "sibylline-scurl[prompt-defender]"
```

## Usage

```bash
# Fetch a URL and extract clean markdown from HTML
scurl https://example.com

# Raw output (disable response middleware)
scurl --raw https://example.com

# Extract article content only (strips nav, ads, sidebars)
scurl --readability https://example.com

# Enable prompt injection detection
scurl --enable prompt-defender https://example.com

# Multilingual prompt injection detection (all 13 supported languages)
scurl --enable prompt-defender --injection-languages all https://example.com

# All curl flags work
scurl -H "Accept: application/json" https://api.example.com/data
```

## Features

- **SecretDefender**: Automatically detects and blocks requests containing exposed secrets/tokens
- **HTML to Markdown**: Converts HTML responses to clean markdown (use `--readability` for article extraction)
- **Prompt Injection Detection**: Detects and handles prompt injection attacks in web content
- **Multilingual Support**: Prompt injection detection in 13 languages
- **Middleware System**: Composable request and response middleware

## Why scurl?

scurl extracts clean, readable content from web pages - perfect for LLM consumption, readability, or bandwidth savings. With prompt injection detection, you can safely fetch web content for AI applications.

### Size Comparison

| Website | curl | scurl | Reduction |
|---------|------|-------|-----------|
| example.com | 513 | 167 | 67.4% |
| news.ycombinator.com | 34,082 | 10,739 | 68.5% |
| en.wikipedia.org/wiki/Curl | 110,373 | 10,044 | 90.9% |
| github.com/anthropics | 296,788 | 353 | 99.9% |
| docs.python.org | 319,554 | 12,348 | 96.1% |

## Prompt Injection Detection

scurl includes a prompt injection detection system that can identify and handle malicious content designed to manipulate LLMs. Enable it with `--enable prompt-defender`.

### Detection System

The prompt defender uses a multi-layer detection approach:

1. **Pattern Matching**: Regex patterns for common injection techniques (instruction override, role injection, system manipulation, jailbreak attempts, etc.)
2. **Motif Analysis**: Fuzzy matching of known injection phrases
3. **ML Classification**: Random Forest classifier with semantic embeddings for novel attack detection

### Supported Languages

Prompt injection detection is available in 13 languages:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ko` | Korean |
| `es` | Spanish | `ru` | Russian |
| `fr` | French | `ar` | Arabic |
| `de` | German | `pt` | Portuguese |
| `zh` | Chinese | `it` | Italian |
| `ja` | Japanese | `hi` | Hindi |
| | | `nl` | Dutch |

Use `--injection-languages` to specify which languages to check:

```bash
# English only (default)
scurl --enable prompt-defender https://example.com

# Specific languages
scurl --enable prompt-defender --injection-languages en,es,fr https://example.com

# All supported languages
scurl --enable prompt-defender --injection-languages all https://example.com
```

### Detection Actions

When a prompt injection is detected, you can configure how scurl handles it:

| Action | Description |
|--------|-------------|
| `warn` | Wrap suspicious spans in `<suspected-prompt-injection>` tags, content unchanged |
| `redact` | Wrap in tags and mask detected patterns with `█` characters (default) |
| `datamark` | Wrap in tags with spotlighting mode for LLM context |
| `metadata` | Add detection metadata to output, content unchanged |
| `silent` | No output modification, detection runs silently |

```bash
# Redact detected injections (default)
scurl --enable prompt-defender --injection-action redact https://example.com

# Just warn, don't modify content
scurl --enable prompt-defender --injection-action warn https://example.com

# Adjust detection threshold (0.0-1.0, default: 0.3)
scurl --enable prompt-defender --injection-threshold 0.5 https://example.com
```

### Custom Pattern Configuration

You can customize or extend the detection patterns by creating YAML configuration files. scurl checks these locations in priority order:

1. **User config**: `~/.config/scurl/patterns/` (highest priority)
2. **Project config**: `.scurl/patterns/` in current directory
3. **Package defaults**: Built-in patterns (fallback)

To override patterns for a language, create a YAML file named `{language_code}.yaml`:

```yaml
# ~/.config/scurl/patterns/en.yaml
language: en
name: English
version: "1.0"

patterns:
  instruction_override:
    - 'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|guidelines?)'
    - 'disregard\s+(everything|all)\s+(above|before|prior)'
    # Add your custom patterns...

  role_injection:
    - 'you\s+are\s+now'
    - 'from\s+now\s+on'
    # ...

motifs:
  instruction_override:
    - 'ignore all instructions'
    - 'forget everything above'
    # Simple phrases for fuzzy matching...
```

Pattern categories:
- `instruction_override`: Attempts to override system instructions
- `role_injection`: Attempts to change the AI's role/persona
- `system_manipulation`: Attempts to enable "developer mode", bypass safety, etc.
- `prompt_leak`: Attempts to extract system prompts
- `jailbreak_keywords`: Known jailbreak techniques (DAN, etc.)
- `encoding_markers`: Base64, ROT13, and other encoding attempts
- `suspicious_delimiters`: Fake system/instruction tags

For CJK languages (Chinese, Japanese, Korean) that don't use spaces between words, add:

```yaml
settings:
  word_boundaries: false
```

## Flags

| Flag | Description |
|------|-------------|
| `--raw` | Disable all response middleware (raw HTML output) |
| `--readability` | Extract article content only (strips nav, ads, sidebars) |
| `--render` | Use headless browser for JS-rendered pages |
| `--disable <slug>` | Disable a middleware by slug (can be repeated) |
| `--enable <slug>` | Enable an opt-in middleware (can be repeated) |
| `--list-middleware` | List available middleware and their slugs |

### Prompt Defender Flags

| Flag | Description |
|------|-------------|
| `--injection-threshold <0.0-1.0>` | Detection sensitivity (default: 0.3) |
| `--injection-action <action>` | Action on detection: `warn`, `redact`, `datamark`, `metadata`, `silent` |
| `--injection-languages <langs>` | Languages to check: comma-separated codes or `all` |

## Middleware Slugs

| Slug | Type | Description |
|------|------|-------------|
| `secret-defender` | Request | Detects and blocks requests containing secrets |
| `readability` | Response | Extracts clean markdown from HTML |
| `prompt-defender` | Response | Detects prompt injection in web content (opt-in) |

## Python API

```python
from scurl.prompt_defender import PromptInjectionDefender

# Create defender with custom settings
defender = PromptInjectionDefender(
    threshold=0.3,
    action="redact",
    languages=["en", "es", "fr"],
)

# Analyze text directly
from scurl.prompt_defender.middleware import PromptInjectionMiddleware
middleware = PromptInjectionMiddleware(languages=["all"])
result = middleware.analyze("Ignore all previous instructions and...")
print(result.is_injection)  # True
print(result.confidence)     # 0.95
print(result.detected_spans) # List of detected injection spans
```

## License

Copyright 2026 [Sibylline Software](https://sibylline.dev)

MIT
