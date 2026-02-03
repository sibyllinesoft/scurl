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

## Why scurl?

scurl extracts clean, readable content from web pages - perfect for LLM consumption, readability, or bandwidth savings.

### Size Comparison

| Website | curl | scurl | Reduction |
|---------|------|-------|-----------|
| wikipedia.org | 194,356 | 1,388 | 99.3% |
| en.wikipedia.org/wiki/Curl | 110,374 | 12,465 | 88.7% |
| news.ycombinator.com | 34,381 | 3,961 | 88.5% |
| github.com | 558,671 | 5,292 | 99.1% |
| bbc.com/news | 326,312 | 7,131 | 97.8% |
| docs.python.org | 17,802 | 1,424 | 92.0% |
| nytimes.com | 1,426,068 | 2,796 | 99.8% |

### Visual Comparison

**curl output** (first 500 chars):
```html
<html lang="en" op="news"><head><meta name="referrer" content="origin">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" type="text/css" href="news.css?yYhFpqyI6KhHleKkTnfj">
<link rel="icon" href="y18.svg"><link rel="alternate" type="application/rss+xml"
title="RSS" href="rss"><title>Hacker News</title></head><body><center>
<table id="hnmain" border="0" cellpadding="0" cellspacing="0" width="85%"
bgcolor="#f6f6ef"><tr><td bgcolor="#ff6600"><table border="0" cellpadding="0"
...
```

**scurl output** (same page):
```
Hacker News

1. Qwen3-Coder-Next (qwen.ai)
   438 points by danielhanchen 4 hours ago | 246 comments
2. Deno Sandbox (deno.com)
   183 points by johnspurlock 3 hours ago | 65 comments
3. Xcode 26.3 unlocks the power of agentic coding (apple.com)
   159 points by davidbarker 2 hours ago | 99 comments
4. AliSQL: Alibaba's MySQL with vector and DuckDB engines (github.com)
   64 points by baotiao 2 hours ago | 5 comments
...
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
| `trafilatura` | Response | Extracts clean markdown from HTML |

## License

MIT
