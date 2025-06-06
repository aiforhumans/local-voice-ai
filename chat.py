
import requests

LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

def chat_with_lmstudio(prompt):
    payload = {
        "model": "llama3",  # Adjust your model name here
        "messages": [
            {"role": "system", "content": "You are a friendly AI companion. Answer warmly."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    response = requests.post(LM_STUDIO_API_URL, json=payload)
    reply = response.json()['choices'][0]['message']['content']
    return reply
