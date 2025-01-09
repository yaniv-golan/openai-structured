# examples/advanced_usage.py
import logging
import os

from openai import OpenAI
from pydantic import BaseModel

from openai_structured.client import openai_structured_call

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ProductInfo(BaseModel):
    name: str
    price: float
    description: str


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

user_prompt = "Tell me about a new laptop."
system_prompt = (
    "Extract product information including name, price, and a brief description."
)

try:
    product_info = openai_structured_call(
        client=client,
        model="gpt-4o-2024-08-06",
        output_schema=ProductInfo,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        logger=logger,  # Pass the logger
    )
    print(product_info)
except Exception as e:
    logger.error("An error occurred:", exc_info=True)
