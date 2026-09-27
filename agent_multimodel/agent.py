import asyncio
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

import logging
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

MODEL_OLLAMA = "ollama_chat/llama3.2"


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city: Name of the city, e.g. "New York".

    Returns:
        dict: status and report, or status and error_message.
    """
    if city.lower() == "new york":
        return {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."}
    return {"status": "error", "error_message": f"Weather information for '{city}' is not available."}


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
    print(f"\n>>> User Query: {query}")
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.actions.state_delta.get('escalation_reason', 'No specific message.')}"
            break

    print(f"<<< Agent Response: {final_response_text}")


async def main():
    weather_agent_ollama = Agent(
        name="weather_agent_ollama",
        model=LiteLlm(model=MODEL_OLLAMA),
        description="Provides weather information (using local Llama 3.2).",
        instruction="You are a helpful weather assistant powered by a local Llama 3.2 model. "
                    "Use the 'get_weather' tool for city weather requests. "
                    "Clearly present successful reports or polite error messages.",
        tools=[get_weather],
    )
    print(f"Agent '{weather_agent_ollama.name}' created using local Ollama model.")

    session_service_ollama = InMemorySessionService()
    APP_NAME, USER_ID, SESSION_ID = "weather_tutorial_app_ollama", "user_1_ollama", "session_001_ollama"

    await session_service_ollama.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner_ollama = Runner(agent=weather_agent_ollama, app_name=APP_NAME, session_service=session_service_ollama)

    print("\n--- Testing Ollama Agent ---")
    await call_agent_async(query="What's the weather in New York?", runner=runner_ollama,
                            user_id=USER_ID, session_id=SESSION_ID)
    await call_agent_async(query="What's the weather in London?", runner=runner_ollama,
                            user_id=USER_ID, session_id=SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())