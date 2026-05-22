"""
llm_provider.py
═════════════════════════════════════════════════════════════════════
Unified LLM interface — same function call, different backend.

WHY THIS FILE EXISTS:
    Without this: every other file has "if openai... elif groq... elif bedrock..."
    With this: every other file just calls llm_chat() and forgets about providers
    To switch providers: change LLM_PROVIDER in config.py, done.

PATTERN — this is called the "Adapter pattern" or "Strategy pattern":
    The rest of the app talks to ONE interface (llm_chat).
    This file translates that to whichever provider is active.

HOW TO ADD A NEW PROVIDER:
    Add a function _xxx_chat() that takes messages + returns (text, tokens)
    Add an "elif" branch to llm_chat()
    Add the model names to config.py
    Done — no other file needs to change
═════════════════════════════════════════════════════════════════════
"""

import os
import json
from typing import List, Tuple

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    AWS_REGION,
    OPENAI_MAIN_MODEL, OPENAI_FAST_MODEL,
    GROQ_MAIN_MODEL,   GROQ_FAST_MODEL,
    BEDROCK_MAIN_MODEL, BEDROCK_FAST_MODEL,
)


# ═════════════════════════════════════════════════════════════════
#  CLIENT INITIALIZATION — only the active provider is set up
# ═════════════════════════════════════════════════════════════════
_openai_client = None
_groq_client   = None
_bedrock_client = None

if LLM_PROVIDER == "openai":
    from openai import AsyncOpenAI
    _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print(f"[LLM] Using OpenAI ({OPENAI_MAIN_MODEL})")

elif LLM_PROVIDER == "groq":
    # Groq uses the OpenAI client interface — just different base URL
    # WHY: Groq deliberately mimics OpenAI's API so you don't rewrite code
    from openai import AsyncOpenAI
    _groq_client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    print(f"[LLM] Using Groq ({GROQ_MAIN_MODEL}) — FREE tier")

elif LLM_PROVIDER == "bedrock":
    # AWS Bedrock uses boto3 (AWS SDK)
    # boto3 reads AWS credentials from environment automatically
    import boto3
    _bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
    )
    print(f"[LLM] Using AWS Bedrock ({BEDROCK_MAIN_MODEL})")

else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ═════════════════════════════════════════════════════════════════
#  PROVIDER-SPECIFIC FUNCTIONS
# ═════════════════════════════════════════════════════════════════

async def _openai_chat(
    messages: List[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, int]:
    """Call OpenAI Chat Completions API."""
    response = await _openai_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text   = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens
    return text, tokens


async def _groq_chat(
    messages: List[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, int]:
    """
    Call Groq API.
    Same shape as OpenAI because Groq uses the OpenAI client.
    """
    response = await _groq_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text   = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens
    return text, tokens


async def _bedrock_chat(
    messages: List[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, int]:
    """
    Call AWS Bedrock — uses Anthropic Claude format.

    NOTE: Bedrock boto3 is sync. We wrap in asyncio.to_thread to keep async.
    """
    import asyncio

    # Bedrock requires system message separated from user/assistant turns
    system_text = ""
    convo = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            convo.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            })

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": convo,
    }
    if system_text:
        body["system"] = system_text

    def call():
        response = _bedrock_client.invoke_model(
            modelId=model,
            body=json.dumps(body),
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        text   = result["content"][0]["text"].strip()
        tokens = result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
        return text, tokens

    return await asyncio.to_thread(call)


# ═════════════════════════════════════════════════════════════════
#  PUBLIC API — what the rest of the app calls
# ═════════════════════════════════════════════════════════════════

def get_main_model() -> str:
    """Return the 'big' model name for the active provider."""
    return {
        "openai":  OPENAI_MAIN_MODEL,
        "groq":    GROQ_MAIN_MODEL,
        "bedrock": BEDROCK_MAIN_MODEL,
    }[LLM_PROVIDER]


def get_fast_model() -> str:
    """Return the 'small/fast' model name for the active provider."""
    return {
        "openai":  OPENAI_FAST_MODEL,
        "groq":    GROQ_FAST_MODEL,
        "bedrock": BEDROCK_FAST_MODEL,
    }[LLM_PROVIDER]


async def llm_chat(
    messages: List[dict],
    model: str = None,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> Tuple[str, int]:
    """
    THE main function every other file uses to talk to an LLM.

    Same signature regardless of which provider is active.

    Args:
        messages:    OpenAI-style [{"role": "user", "content": "..."}] list
        model:       Specific model name, or None to use the main model
        max_tokens:  Cap on response length
        temperature: 0 = deterministic, 1 = creative

    Returns:
        (response_text, total_tokens_used)

    WHAT IF YOU DIDN'T HAVE THIS WRAPPER:
        Every file would have provider-specific code.
        Switching providers would mean changing 20 places.
        Adding a new provider would touch every file.
    """
    if model is None:
        model = get_main_model()

    if LLM_PROVIDER == "openai":
        return await _openai_chat(messages, model, max_tokens, temperature)
    elif LLM_PROVIDER == "groq":
        return await _groq_chat(messages, model, max_tokens, temperature)
    elif LLM_PROVIDER == "bedrock":
        return await _bedrock_chat(messages, model, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown provider: {LLM_PROVIDER}")
