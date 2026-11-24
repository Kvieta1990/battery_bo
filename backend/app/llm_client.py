"""
LLM access for the agents that genuinely need language understanding: Condition Processor
(reading free-text lab notebook entries) and Encoding Determination (proposing how to encode
an ambiguous numeric-looking-but-categorical parameter). Both call `structured_call()`, which
wraps LangChain's `.with_structured_output()` so the model's answer is always validated
against a Pydantic schema instead of hand-parsed free text.

Deliberately NOT used for the Bayesian-optimization math itself (see core/bayes_opt_core.py's
module docstring for why) -- only for extraction and natural-language summaries.
"""
from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)

_model_cache = {}


class LLMNotConfigured(RuntimeError):
    pass


def get_chat_model(temperature: float = 0.0):
    key = (config.LLM_PROVIDER, temperature)
    if key in _model_cache:
        return _model_cache[key]

    if config.LLM_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            raise LLMNotConfigured("OPENAI_API_KEY is not set. Add it to backend/.env and restart the server.")
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY, temperature=temperature)
    else:
        if not config.ANTHROPIC_API_KEY:
            raise LLMNotConfigured("ANTHROPIC_API_KEY is not set. Add it to backend/.env and restart the server.")
        from langchain_anthropic import ChatAnthropic
        # The installed anthropic SDK (2.x) talks to an API version that rejects `temperature`
        # (and top_p/top_k) outright for every current Claude model -- 400 "Extra inputs are
        # not permitted". Don't send it; the model runs at its default sampling instead.
        model = ChatAnthropic(model=config.ANTHROPIC_MODEL, api_key=config.ANTHROPIC_API_KEY,
                               max_tokens=4096)
    _model_cache[key] = model
    return model


def structured_call(system: str, user: str, schema: Type[T], temperature: float = 0.0) -> T:
    """Call the configured LLM and parse its answer into `schema`. Raises LLMNotConfigured if
    no API key is set, and whatever LangChain/the provider SDK raises on a request failure --
    callers should catch and surface that as a per-sample flag rather than let one bad call
    kill a whole batch."""
    model = get_chat_model(temperature=temperature)
    structured = model.with_structured_output(schema)
    messages = [("system", system), ("human", user)]
    result = structured.invoke(messages)
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def plain_call(system: str, user: str, temperature: float = 0.0) -> str:
    model = get_chat_model(temperature=temperature)
    messages = [("system", system), ("human", user)]
    return model.invoke(messages).content
