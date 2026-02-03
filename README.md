# scurl

[![PyPI version](https://badge.fury.io/py/scurl.svg)](https://badge.fury.io/py/scurl)
[![CI](https://github.com/yourusername/scurl/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/scurl/actions/workflows/ci.yml)

A secure curl wrapper with middleware support and HTML-to-markdown extraction.

## Installation

```bash
pip install scurl
```

Or with [pipx](https://pipx.pypa.io/) (recommended for CLI tools):

```bash
pipx install scurl
```

## Usage

```bash
# Fetch a URL and extract clean markdown from HTML
scurl https://example.com

# Raw output (disable response middleware)
scurl --raw https://example.com

# All curl flags work
scurl -H "Accept: application/json" https://api.example.com/data
```

## Features

- **SecretDefender**: Automatically detects and blocks requests containing exposed secrets/tokens
- **TrafilaturaExtractor**: Extracts clean markdown from HTML responses
- **Middleware System**: Composable request and response middleware

## Flags

| Flag | Description |
|------|-------------|
| `--raw` | Disable all response middleware |
| `--disable <slug>` | Disable a middleware by slug (can be repeated) |
| `--enable <slug>` | Override a middleware's block (can be repeated) |
| `--list-middleware` | List available middleware and their slugs |

## Middleware Slugs

| Slug | Type | Description |
|------|------|-------------|
| `secret-defender` | Request | Detects and blocks requests containing secrets |
| `trafilatura` | Response | Extracts clean markdown from HTML |

## License

MIT
