import os
from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from .agent import weather_agent, root_agent, time_agent
from google.genai import types
import asyncio
from google.adk.models.lite_llm import LiteLlm


# Load environment variables from .env file
load_dotenv()

# Verify API Keys are set
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') and os.environ['OPENAI_API_KEY'] != 'YOUR_OPENAI_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"Anthropic API Key set: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') and os.environ['ANTHROPIC_API_KEY'] != 'YOUR_ANTHROPIC_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")

print("\nEnvironment configured.")

# --- Session Management ---
# Key Concept: SessionService stores conversation history & state.
# InMemorySessionService is simple, non-persistent storage for this tutorial.
session_service = InMemorySessionService()

# Define constants for identifying the interaction context
APP_NAME = "weather_tutorial_app"
USER_ID = "user_1"
SESSION_ID = "session_001"  # Using a fixed ID for simplicity

# Create the specific session where the conversation will happen
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)
print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

# --- Runner ---
# Key Concept: Runner orchestrates the agent execution loop.
weather_runner = Runner(
    agent=weather_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

time_runner = Runner(
    agent=time_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

general_runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

runner_root_stateful = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# Modify call_agent_async to accept a runner
async def call_agent_async(runner: Runner, query: str, user_id: str, session_id: str):
    """Sends a query to the specified agent and prints the final response."""
    print(f"\n>>> User Query to {runner.agent.name}: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:  # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break  # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")

async def run_team_conversation():
    print("\n--- Testing Agent Team Delegation ---")
    
    # Create a NEW session service with state
    session_service = InMemorySessionService()

    # Define session constants
    APP_NAME = "weather_tutorial_agent_team"
    USER_ID = "user_1_agent_team"
    SESSION_ID = "session_001_agent_team"

    # Define initial state for tracking conversation
    initial_state = {
        "conversation_turns": 0,
        "last_query": None
    }
    initial_state["user_preference_temperature_unit"] = "Celsius"
    # Create session with initial state
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )
    print(f"Session created with state: {session.state}")

    runner_agent_team = Runner(
        agent=weather_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    print(f"Runner created for agent '{weather_agent.name}'.")

    # Test conversation with state tracking
    print("\n--- Starting Conversation ---")
    
    # First turn: Greeting
    await call_agent_async(
        runner=runner_agent_team,
        query="Hello there!",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Second turn: Weather query
    await call_agent_async(
        runner=runner_agent_team,
        query="What is the weather in New York?",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Test the guardrail
    await call_agent_async(
        runner=runner_agent_team,
        query="What is the weather in BLOCK city?",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Final turn: Farewell
    await call_agent_async(
        runner=runner_agent_team,
        query="Thanks, bye!",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

async def run_guardrail_test_conversation():
    print("\n--- Testing Model Input Guardrail ---")
    
    # Create a NEW session service with state
    session_service = InMemorySessionService()

    # Define session constants
    APP_NAME = "guardrail_test"
    USER_ID = "user_guardrail_test"
    SESSION_ID = "session_guardrail_test"

    # Define initial state
    initial_state = {
        "user_preference_temperature_unit": "Celsius",
        "guardrail_block_keyword_triggered": False,
        "last_weather_report": None
    }

    # Create session with initial state
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )
    print(f"Session created with state: {session.state}")

    # Create runner for testing
    runner_guardrail = Runner(
        agent=weather_agent,  # Using weather agent which has the guardrail
        app_name=APP_NAME,
        session_service=session_service
    )
    print(f"Runner created for agent '{weather_agent.name}' with guardrail.")

    # Test normal request
    print("\n1. Testing normal request:")
    await call_agent_async(
        runner=runner_guardrail,
        query="What is the weather in London?",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Test blocked keyword
    print("\n2. Testing blocked keyword:")
    await call_agent_async(
        runner=runner_guardrail,
        query="BLOCK the request for weather in Tokyo",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Test normal greeting
    print("\n3. Testing normal greeting:")
    await call_agent_async(
        runner=runner_guardrail,
        query="Hello again",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Test Paris weather request (should be blocked by tool guardrail)
    print("\n4. Testing Paris weather request:")
    await call_agent_async(
        runner=runner_guardrail,
        query="What is the weather in Paris?",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Check final state
    final_session = session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    if final_session:
        print("\n--- Final Session State (After Guardrail Test) ---")
        print(f"Model Guardrail Triggered: {final_session.state.get('guardrail_block_keyword_triggered', False)}")
        print(f"Tool Guardrail Triggered: {final_session.state.get('tool_block_triggered', False)}")
        print(f"Last Weather Report: {final_session.state.get('last_weather_report')}")
        print(f"Temperature Unit: {final_session.state.get('user_preference_temperature_unit')}")
    else:
        print("\n❌ Error: Could not retrieve final session state.")

async def main():
    print("\n=== Starting Tests ===")
    
    # Run the team conversation test
    print("\n=== Running Team Conversation Test ===")
    await run_team_conversation()
    
    # Run the guardrail test
    print("\n=== Running Guardrail Test ===")
    await run_guardrail_test_conversation()
    
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(main()) 