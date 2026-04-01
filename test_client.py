import openai
import os
import sys

# Configuration
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM doesn't require a key by default
)

def test_chat_completion():
    print("Testing Chat Completion...")
    try:
        response = client.chat.completions.create(
            model="qwen2.5-32b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a short joke about AI."}
            ],
            max_tokens=50
        )
        print("Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error connecting to vLLM server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_chat_completion()
