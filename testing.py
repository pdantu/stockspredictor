import pandas as pd
from googlefinance import getQuotes
import yfinance as yf
a = yf.Ticker('AAPL')
print(a.info)
# from openai import OpenAI
# client = OpenAI()
  
# assistant = client.beta.assistants.create(
#   name="Math Tutor",
#   instructions="You are a personal math tutor. Write and run code to answer math questions.",
#   tools=[{"type": "code_interpreter"}],
#   model="gpt-4o",
# )
# thread = client.beta.threads.create()
# message = client.beta.threads.messages.create(
#   thread_id=thread.id,
#   role="user",
#   content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
# )