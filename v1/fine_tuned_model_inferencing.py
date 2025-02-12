from openai import OpenAI
import os

OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')
client = OpenAI(api_key=OPEN_AI_KEY)

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:dami04glorygmailcom::9kImAlxd",
  messages=[
      {"role": "system", "content": "You are an university teaching assistant chatbot that guides students"}, 
      {"role": "user", "content": "I'm so sorry I'm a fool I somehow thought we only needed 2 instead of 3 critiques for 4740. Could I submit c6 late now and take a 7 letter grade penalty? Alternatively, since c6 was already discussed, I could critique another paper:\n\nex.\n\n1. Machine-to-corpus-to-machine training regime: https://arxiv.org/pdf/2110.07178.pdf\n\n2. Character-level language model: https://arxiv.org/pdf/1508.06615.pdf\n\n3. Fixing bottleneck of fixed-length hidden in NMT translation: https://arxiv.org/pdf/1409.0473.pdf \n\n"}
    ]
)
print(completion.choices[0].message.content)

#{"role": "assistant", "content": "This has been resolved."}
# {"role": "assistant", "content": "This is very cool!   Are you in touch with our Digital Humanities faculty (in CIS and more broadly)?  The main CSy types are David Mimno and Matt Wilkens.  Both are in the Information Science department.\n\nhttps://cornell.campusgroups.com/dhc/home/\n\nhttps://digitalscholarship.library.cornell.edu/\n\nThe latter link indicates that there was a summer graduate fellowship at least for last summer of 2021."}]}
# {"role": "assistant", "content": "Thanks for the support \u2764\ufe0f \u2764\ufe0f \n\n"}]}
# {"role": "assistant", "content": "I will just give you the average of your two critiques as the grade for the 3rd critique."}]}

