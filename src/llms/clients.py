from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

import json
import logging
import os
import re
import sys
import threading
import time
import traceback

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
for _p in (SRC_ROOT, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import AGENT_LLM_MODEL, API_KEY, BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    url: str = BASE_URL
    model_name: str = AGENT_LLM_MODEL
    api_key: str = API_KEY
    max_tokens: int = 32768
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: int = 0
    timeout: int = 200
    retry_attempts: int = 20
    think_bool: bool = False
    _client: Optional[Any] = field(default=None, repr=False)

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.url)
        return self._client


class LLMStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.agent_calls = 0
        self.judge_calls = 0
        self.total_latency = 0.0

    def record(self, kind: str, latency: float):
        with self._lock:
            if kind == "agent":
                self.agent_calls += 1
            else:
                self.judge_calls += 1
            self.total_latency += latency

    def reset(self):
        with self._lock:
            self.agent_calls = self.judge_calls = 0
            self.total_latency = 0.0

    def snapshot(self):
        with self._lock:
            return {
                "agent_calls": self.agent_calls,
                "judge_calls": self.judge_calls,
                "total_latency": round(self.total_latency, 2),
            }


stats = LLMStats()


def _strip_think(text: str):
    if text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _format_messages(messages: Union[str, List[Dict[str, str]]]):
    if isinstance(messages, str):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages},
        ]
    return messages


def get_from_llm(
    messages: Union[str, List[Dict[str, str]]],
    model_name: str = AGENT_LLM_MODEL,
    **kwargs,
):
    config = ModelConfig(model_name=model_name)
    formatted_messages = _format_messages(messages)

    temperature = kwargs.get("temperature", config.temperature)
    top_p = kwargs.get("top_p", config.top_p)
    max_tokens = kwargs.get("max_tokens", config.max_tokens)

    for attempt in range(config.retry_attempts):
        try:
            response = config.client.chat.completions.create(
                model=config.model_name,
                messages=formatted_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "enable_thinking": config.think_bool,
                    "top_k": config.top_k,
                    "min_p": config.min_p,
                },
            )
            text = response.choices[0].message.content
            if text:
                return _strip_think(text)
            logger.error("Empty response from %s", model_name)
            time.sleep(5)
        except Exception as e:
            logger.warning("Attempt %s failed for %s: %s", attempt + 1, model_name, e)
            if attempt == config.retry_attempts - 1:
                logger.error(traceback.format_exc())
            else:
                time.sleep(min(5, 1 + attempt))
    return None


def agent_chat(prompt: str):
    t0 = time.time()
    response = get_from_llm(prompt)
    stats.record("agent", time.time() - t0)
    return response or ""


def _extract_json(text: str):
    if text is None:
        return None
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            idx_start = text.find(start_char)
            idx_end = text.rfind(end_char)
            if idx_start != -1 and idx_end > idx_start:
                try:
                    return json.loads(text[idx_start : idx_end + 1])
                except json.JSONDecodeError:
                    continue
        return None


def agent_chat_json(prompt: str, retries: int = 2):
    for _ in range(retries + 1):
        response = agent_chat(prompt)
        result = _extract_json(response)
        if result is not None:
            return result
    return None
