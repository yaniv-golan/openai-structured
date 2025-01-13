"""Basic async example for openai-structured."""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from openai_structured import async_openai_structured_call


class WeatherForecast(BaseModel):
    """A structured weather forecast."""

    temperature: float = Field(..., description="Temperature in Celsius")
    conditions: str = Field(..., pattern="^(sunny|cloudy|rainy|snowy)$")
    wind_speed: float = Field(..., ge=0, description="Wind speed in km/h")
    humidity: int = Field(..., ge=0, le=100, description="Humidity percentage")


async def main():
    """Run the basic async example."""
    client = AsyncOpenAI()

    # Make a structured call to get weather forecast
    forecast = await async_openai_structured_call(
        client=client,
        model="gpt-4o-2024-08-06",
        system_prompt="You are a weather forecasting assistant.",
        user_prompt="What's the weather forecast for Tokyo today?",
        output_schema=WeatherForecast,
    )

    print("\nWeather Forecast for Tokyo:")
    print(f"Temperature: {forecast.temperature}°C")
    print(f"Conditions: {forecast.conditions}")
    print(f"Wind Speed: {forecast.wind_speed} km/h")
    print(f"Humidity: {forecast.humidity}%")


if __name__ == "__main__":
    asyncio.run(main())
