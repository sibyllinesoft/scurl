# scurl

[![PyPI version](https://badge.fury.io/py/scurl.svg)](https://badge.fury.io/py/scurl)
[![CI](https://github.com/yourusername/scurl/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/scurl/actions/workflows/ci.yml)

A secure curl wrapper with middleware support and HTML-to-markdown extraction.

## Installation

```bash
pip install sibylline-scurl
```

Or with [pipx](https://pipx.pypa.io/) (recommended for CLI tools):

```bash
pipx install sibylline-scurl
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
- **ReadabilityExtractor**: Extracts clean markdown from HTML responses using readability + html2text
- **Middleware System**: Composable request and response middleware

## Why scurl?

scurl extracts clean, readable content from web pages - perfect for LLM consumption, readability, or bandwidth savings.

### Size Comparison

| Website | curl | scurl | Reduction |
|---------|------|-------|-----------|
| example.com | 513 | 167 | 67.4% |
| news.ycombinator.com | 34,082 | 10,739 | 68.5% |
| en.wikipedia.org/wiki/Curl | 110,373 | 10,044 | 90.9% |
| github.com/anthropics | 296,788 | 353 | 99.9% |
| docs.python.org | 319,554 | 12,348 | 96.1% |

### Visual Comparison

**curl output** (Wikipedia, first 500 chars):
```html
<!DOCTYPE html><html class="client-nojs" lang="en" dir="ltr"><head>
<meta charset="UTF-8"/><title>Curl (programming language) - Wikipedia</title>
<script>(function(){var className="client-js";var cookie=document.cookie.
match(/(?:^|; )enwikimwclientpreferences=([^;]+)/);if(cookie){cookie[1].
split('%2C').forEach(function(pref){className=className.replace(new
RegExp('(^| )'+pref.replace(/-hierarchical-hierarchical/,'')
+'($| )'),'$1teleported-hierarchical$2');});...
```

**scurl output** (same page):
```markdown
# Curl (programming language) - Wikipedia

**Curl** is a reflective object-oriented programming language for interactive
web applications, whose goal is to provide a smoother transition between
content formatting and computer programming. It makes it possible to embed
complex objects in simple documents without needing to switch between
programming languages or development platforms.

The Curl implementation initially consisted of an interpreter only; a compiler
was added later...
```

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
| `readability` | Response | Extracts clean markdown from HTML |

## License

Copyright 2026 [Sibylline Software](https://sibylline.dev)

MIT
