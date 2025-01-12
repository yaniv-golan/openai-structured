# examples/basic_call.py
from openai import OpenAI
from pydantic import BaseModel

from openai_structured import openai_structured_call


class UserInfo(BaseModel):
    name: str
    age: int


def main():
    client = OpenAI()
    result = openai_structured_call(
        client=client,
        model="gpt-4o-2024-08-06",
        output_schema=UserInfo,
        user_prompt="Tell me about John who is 30 years old",
        system_prompt="Extract user information",
    )
    print(f"Name: {result.name}, Age: {result.age}")


if __name__ == "__main__":
    main()
