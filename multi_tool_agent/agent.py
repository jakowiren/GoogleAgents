from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from typing import Optional
from zoneinfo import ZoneInfo
from .constants import MODEL_GEMINI_2_0_FLASH, MODEL_GPT_4O, MODEL_CLAUDE_SONNET
from .tools import get_weather, get_current_time, say_goodbye, say_hello
import warnings
from dotenv import load_dotenv
import logging
from .tools import block_paris_tool_guardrail
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Load environment variables
load_dotenv()

def block_keyword_guardrail(
    callback_context: CallbackContext, 
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Blocks requests containing the keyword 'BLOCK'."""
    agent_name = callback_context.agent_name
    print(f"--- Callback: block_keyword_guardrail running for agent: {agent_name} ---")

    # Get last user message
    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts:
                if content.parts[0].text:
                    last_user_message_text = content.parts[0].text
                    break

    print(f"--- Callback: Inspecting message: '{last_user_message_text[:100]}...' ---")

    # Check for blocked keyword
    keyword_to_block = "BLOCK"
    if keyword_to_block in last_user_message_text.upper():
        print(f"--- Callback: Found '{keyword_to_block}'. Blocking LLM call! ---")
        callback_context.state["guardrail_block_keyword_triggered"] = True
        
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"I cannot process this request because it contains the blocked keyword '{keyword_to_block}'.")],
            )
        )
    
    print(f"--- Callback: Keyword not found. Allowing LLM call for {agent_name}. ---")
    return None

# Define specialized agents with different models
time_agent = Agent(
    name="time_agent_v1",
    model=LiteLlm(model=MODEL_GPT_4O),
    description="Provides accurate time information for cities.",
    instruction="You are a time zone specialist. Help users get accurate time information for different cities.",
    tools=[get_current_time],
    before_model_callback=block_keyword_guardrail,
)

# Create greeting agents for each parent
def create_greeting_agent(name_suffix=""):
    return Agent(
        model=LiteLlm(model=MODEL_GPT_4O),
        name=f"greeting_agent{name_suffix}",
        instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
                    "Use the 'say_hello' tool to generate the greeting. "
                    "If the user provides their name, make sure to pass it to the tool. "
                    "Do not engage in any other conversation or tasks.",
        description="Handles simple greetings and hellos using the 'say_hello' tool.",
        tools=[say_hello],
        before_model_callback=block_keyword_guardrail,
    )

# Create farewell agents for each parent
def create_farewell_agent(name_suffix=""):
    return Agent(
        model=LiteLlm(model=MODEL_GPT_4O),
        name=f"farewell_agent{name_suffix}",
        instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
                    "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation "
                    "(e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
                    "Do not perform any other actions.",
        description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.",
        tools=[say_goodbye],
        before_model_callback=block_keyword_guardrail,
    )

# Create root agent with its own sub-agents
root_agent = Agent(
    name="weather_time_agent",
    model=LiteLlm(model=MODEL_CLAUDE_SONNET),
    description="Agent to answer questions about the time and weather in a city.",
    instruction="I can answer your questions about the time and weather in a city.",
    tools=[get_weather, get_current_time],
    sub_agents=[create_greeting_agent("_root"), create_farewell_agent("_root")],
    output_key="last_weather_report",
    before_model_callback=block_keyword_guardrail,
)

# Create weather agent with its own sub-agents
weather_agent = Agent(
    name="weather_agent_v1",
    model=LiteLlm(model=MODEL_GPT_4O),
    description="Provides weather information for specific cities.",
    instruction="You are a helpful weather assistant. Your primary goal is to provide current weather reports. "
                "When the user asks for the weather in a specific city, "
                "you MUST use the 'get_weather' tool to find the information. "
                "Analyze the tool's response: if the status is 'error', inform the user politely about the error message. "
                "If the status is 'success', present the weather 'report' clearly and concisely to the user. "
                "Only use the tool when a city is mentioned for a weather request.",
    tools=[get_weather],
    sub_agents=[create_greeting_agent("_weather"), create_farewell_agent("_weather")],
    before_model_callback=block_keyword_guardrail,
    before_tool_callback=block_paris_tool_guardrail,
)
