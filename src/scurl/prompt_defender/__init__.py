"""Prompt injection detection middleware for scurl.

Detection logic is provided by sibylline-clean.
This module provides the scurl middleware adapter.
"""

from .middleware import PromptInjectionDefender

__all__ = [
    "PromptInjectionDefender",
]
