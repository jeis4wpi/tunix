# model_clients.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

@dataclass
class GenerationConfig:
   max_new_tokens: int = 256
   temperature: float = 0.7
   top_p: float = 0.95
   top_k: Optional[int] = None
   stop: Optional[List[str]] = None
   presence_penalty: float = 0.0
   frequency_penalty: float = 0.0

@dataclass
class GenerateResult:
   text: str
   finish_reason: str = "stop"
   usage: Optional[Dict[str, Any]] = None
   raw: Optional[Any] = None

class ModelClient(Protocol):
   def generate(self, messages: List[Dict[str, str]], config: Optional[GenerationConfig] = None) -> GenerateResult: ...
   async def agenerate(self, messages: List[Dict[str, str]], config: Optional[GenerationConfig] = None) -> GenerateResult: ...