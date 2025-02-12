from openai import OpenAI
import os

OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')
client = OpenAI(api_key=OPEN_AI_KEY)

client.fine_tuning.jobs.create(
  training_file="file-RGTZwWrOE3bTwbRQbkeec7ot", 
  validation_file="file-wCKVkTYx23kwM3weZCSiYKFO",
  model="gpt-3.5-turbo"
)