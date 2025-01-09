# examples/basic_call.py
import os
from typing import List

from openai import OpenAI
from pydantic import BaseModel

from openai_structured.client import openai_structured_call


class UserInfo(BaseModel):
    name: str
    age: int


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

user_prompt = "Tell me about a user named John who is 30 years old."
system_prompt = "Extract user information."

try:
    user_info = openai_structured_call(
        client=client,
        model="gpt-4o-2024-08-06",
        output_schema=UserInfo,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
    print(user_info)
except Exception as e:
    print(f"Error: {e}")
