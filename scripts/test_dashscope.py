import os
from http import HTTPStatus

import requests


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "glm-5.1"


def request_qwen36_plus(prompt):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Missing DASHSCOPE_API_KEY environment variable")

    url = f"{BASE_URL}/chat/completions"

    if isinstance(prompt, list):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += [
            {"role": turn.get("role", "user"), "content": turn.get("content", "")}
            for turn in prompt
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(prompt)},
        ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"http status @ {resp.status_code}, body={resp.text}")

    data = resp.json() if resp.content else {}
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


if __name__ == "__main__":
    text = request_qwen36_plus("请用一句话介绍你自己")
    print(text)


