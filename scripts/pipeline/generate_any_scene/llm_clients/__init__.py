"""
LLM Clients for Enhanced Caption Generation
"""

from .base_client import LLMClient
from .vllm_client import VLLMClient

__all__ = ['LLMClient', 'VLLMClient']
