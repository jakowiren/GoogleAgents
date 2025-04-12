import os
from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from .agent import root_agent
from google.genai import types
import asyncio
import time

# Load environment variables
load_dotenv()

def check_api_keys():
    """Check if required API keys are set."""
    print("API Keys Set:")
    print(f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') and os.environ['OPENAI_API_KEY'] != 'YOUR_OPENAI_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
    print(f"GitHub Token set: {'Yes' if os.environ.get('GITHUB_TOKEN') and os.environ['GITHUB_TOKEN'] != 'YOUR_GITHUB_TOKEN' else 'No (Optional for public repos)'}")
    print("\nEnvironment configured.")

async def call_agent_async(runner: Runner, query: str, user_id: str, session_id: str):
    """Sends a query to the specified agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    print(f"<<< Agent Response: {final_response_text}\n")
    # Add a small delay between queries to avoid rate limits
    await asyncio.sleep(1)
    return final_response_text

async def test_workflow():
    """Test the complete workflow demonstrating fetch, analysis, and metadata capabilities."""
    print("\n=== Starting Multi-Agent Workflow Test ===")
    
    # Create session service and session
    session_service = InMemorySessionService()
    
    # Define session constants
    APP_NAME = "multi_agent_workflow"
    USER_ID = "test_user_1"
    SESSION_ID = f"test_session_{int(time.time())}"
    
    # Create the session
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    # Create root agent runner
    root_runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Test queries organized by agent type
    test_scenarios = [
        {
            "phase": "Initial State",
            "queries": [
                "Show me the last commit message",  # Should prompt to fetch first
                "How does the code work?",  # Should prompt to fetch first
            ]
        },
        {
            "phase": "Repository Fetching",
            "queries": [
                "https://github.com/jakowiren/GoogleAgents",  # Fetch repository
            ]
        },
        {
            "phase": "Metadata Analysis",
            "queries": [
                "Tell me about the last commit in the GoogleAgents repository",  # Tests commit info with explicit repo name
                "What branch is the current code on?",  # Tests branch info using state
                "List all tags in the repository",  # Tests tag info
                "Give me a complete overview of the repository's current state",  # Tests general metadata
            ]
        },
        {
            "phase": "Code Analysis",
            "queries": [
                "How does the code handle GitHub API authentication?",  # Tests code search
                "Explain how the vectorstore is created and used",  # Tests implementation details
                "What are the main classes and their responsibilities?",  # Tests code structure
            ]
        },
        {
            "phase": "Mixed Queries",
            "queries": [
                "When was the authentication code last modified?",  # Tests combined code + metadata
                "Show me the implementation of the most recently changed file",  # Tests combined analysis
            ]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n=== Testing {scenario['phase']} ===")
        for query in scenario['queries']:
            print(f"\n--- Processing Query: {query} ---")
            await call_agent_async(root_runner, query, USER_ID, SESSION_ID)
            # Add a delay between scenarios to avoid rate limits
            await asyncio.sleep(2)
        # Add a longer delay between phases
        await asyncio.sleep(5)

async def main():
    """Main entry point for testing the agents."""
    print("\n=== Starting Multi-Agent System Test ===")
    print("This test will demonstrate the interaction between three specialized agents:")
    print("1. Fetch Agent: Downloads and prepares repositories")
    print("2. Metadata Agent: Provides Git history and repository information")
    print("3. Analysis Agent: Examines code implementation and patterns")
    
    # Check environment setup
    check_api_keys()
    
    try:
        # Run the workflow test
        await test_workflow()
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    asyncio.run(main())