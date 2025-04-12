import os
from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from .agent import root_agent
from google.genai import types
import asyncio
import time
import json

# Load environment variables
load_dotenv()

def check_api_keys():
    """Check if required API keys are set."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    if not github_token:
        print("Warning: GITHUB_TOKEN not found. Only public repositories will be accessible.")

async def call_agent_async(runner: Runner, query: str) -> str:
    """Send a query to the agent and print the final response."""
    response = await runner.arun(query)
    print(f"\nQuery: {query}")
    print(f"Response: {response}\n")
    print("-" * 80)
    return response

async def test_repo_analysis_workflow():
    """Test the complete workflow of fetching, analyzing, and reporting on a repository."""
    # Create session service and session
    session_service = InMemorySessionService()
    APP_NAME = "google_agents"
    USER_ID = "test_user"
    SESSION_ID = "test_session"
    session = session_service.create_session(APP_NAME, USER_ID, SESSION_ID)
    
    # Create runner with session
    runner = Runner(root_agent, session=session)
    
    # Test scenarios organized by phases
    test_scenarios = {
        "Initial State": [
            "What can you help me with?",
        ],
        "Repository Fetching": [
            "Can you fetch this repository: https://github.com/jakowiren/GoogleAgents",
        ],
        "Metadata Analysis": [
            "What was the last commit to the repository?",
            "Show me the commit history for the tools.py file",
        ],
        "Code Analysis": [
            "How is the code organized in this repository?",
            "What are the main components and their relationships?",
            "How is error handling implemented in the tools?",
        ],
        "Report Generation": [
            "Generate a comprehensive report analyzing the code architecture and implementation patterns",
            "Create a focused report about the error handling and testing approaches in the codebase",
            "Generate a report summarizing recent development history and code evolution",
        ],
        "Mixed Queries": [
            "What design patterns are used in the agent implementation?",
            "How has the error handling evolved over time?",
            "Generate a report focusing on code quality and potential improvements",
        ]
    }
    
    # Run test scenarios
    for phase, queries in test_scenarios.items():
        print(f"\n=== Testing Phase: {phase} ===\n")
        for query in queries:
            await call_agent_async(runner, query)
            # Add delay between queries to avoid rate limits
            time.sleep(2)
        # Add longer delay between phases
        time.sleep(5)

async def main():
    """Main function to run the test workflow."""
    try:
        # Check environment setup
        check_api_keys()
        
        # Run the complete workflow test
        await test_repo_analysis_workflow()
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())