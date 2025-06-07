"""Interface for chatting with an LM Studio API server."""

from __future__ import annotations

import requests

LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"


def chat_with_lmstudio(prompt: str) -> str:
    """Send a prompt to LM Studio and return the response text."""

    payload = {
        "model": "llama3",  # Replace with your model name
        "messages": [
            {
                "role": "system",
                "content": "You are a friendly AI companion. Answer warmly.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    resp = requests.post(LM_STUDIO_API_URL, json=payload, timeout=60)
    data = resp.json()
    return data["choices"][0]["message"]["content"]
