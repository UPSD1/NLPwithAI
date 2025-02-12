from openai import OpenAI
import os

OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')
client = OpenAI(api_key=OPEN_AI_KEY)

client.files.create(
  file=open("test_dataset.jsonl", "rb"),
  purpose="fine-tune"
)