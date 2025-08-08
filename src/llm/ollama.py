"""
LLM wrapper for Ollama to get responses.
"""

import ollama

MODEL = "llama3.2:3b"


def llm_response(system_prompt: str, user_prompt: str) -> str:
    """
    Send a request to the LLM and return the response.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return response["message"]["content"]