# examples/advanced_usage.py
from openai import OpenAI
from openai_structured import (
    openai_structured_call,
    OpenAIClientError,
)
from pydantic import BaseModel


class ProductInfo(BaseModel):
    name: str
    price: float
    description: str


def main():
    client = OpenAI()
    try:
        result = openai_structured_call(
            client=client,
            model="gpt-4",
            output_schema=ProductInfo,
            user_prompt=(
                "Tell me about a high-end laptop with detailed specifications"
            ),
            system_prompt="Extract product details with exact pricing",
        )
        print(f"{result.name}: ${result.price}")
        print(f"Description: {result.description}")
    except OpenAIClientError as error:
        print(f"Error occurred: {error}")


if __name__ == "__main__":
    main()
