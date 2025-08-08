"""
LLM wrapper for Ollama to send calls.
"""

import ollama

def llm_response(system_prompt: str, user_prompt: str) -> str:
    """
    Send a request to the LLM and return the response.
    """
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return response["message"]["content"]