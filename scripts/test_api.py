from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()  # reads your .env file
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say: API key is working"}],
    max_tokens=20
)
print(response.choices[0].message.content)