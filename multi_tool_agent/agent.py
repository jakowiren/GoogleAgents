from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from typing import Optional
from .constants import MODEL_GEMINI_2_0_FLASH, MODEL_GPT_4O, MODEL_CLAUDE_SONNET
from .tools import (
    fetch_github_repo, 
    analyze_code_with_vectorstore, 
    get_repository_metadata,
    generate_analysis_report
)
import warnings
from dotenv import load_dotenv
import logging
import re
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

# Create GitHub fetch agent
github_fetch_agent = Agent(
    name="github_fetch_agent_v1",
    model=LiteLlm(model=MODEL_GPT_4O),
    description="Downloads GitHub repositories and prepares them for code analysis by creating a vectorstore.",
    instruction="You are a GitHub repository preparation assistant. Your primary goal is to fetch repositories and prepare them for analysis. "
                "When a user provides a GitHub repository URL:\n"
                "1. Use the 'fetch_github_repo' tool to:\n"
                "   * Download the repository\n"
                "   * Create and save a vectorstore of the code\n"
                "2. Analyze the tool's response:\n"
                "   * If status is 'error', inform the user about the error message and suggest fixes\n"
                "   * If status is 'success', confirm that:\n"
                "     - The repository was fetched successfully\n"
                "     - The vectorstore was created and saved\n"
                "     - The repository is ready for analysis\n"
                "3. Let the user know they can now use the analysis agent to explore the code\n"
                "Only use the tool when a valid GitHub repository URL is provided.",
    tools=[fetch_github_repo],
    before_model_callback=block_keyword_guardrail,
    output_key="last_fetched_repo"
)

# Create Git Metadata agent
git_metadata_agent = Agent(
    name="git_metadata_agent_v1",
    model=LiteLlm(model=MODEL_GPT_4O),
    description="Specializes in retrieving and explaining Git repository metadata including commits, branches, and development history.",
    instruction="You are a Git metadata specialist that provides detailed information about repository history and structure. "
                "Your primary goal is to help users understand the development history and state of repositories. "
                "When a user asks about repository metadata:\n"
                "1. Use the 'get_repository_metadata' tool to fetch detailed Git information\n"
                "2. Based on the query type, focus on relevant metadata:\n"
                "   a) For commit queries:\n"
                "      - Show commit hash, message, author, and timestamps\n"
                "      - Explain when and why changes were made\n"
                "   b) For branch queries:\n"
                "      - Display active branch and available remotes\n"
                "      - Explain the repository's branch structure\n"
                "   c) For tag queries:\n"
                "      - List available tags and their purposes\n"
                "   d) For general repository info:\n"
                "      - Provide overview of repository state\n"
                "      - Include fetch time to indicate data freshness\n"
                "3. Format the information clearly:\n"
                "   - Use markdown formatting for readability\n"
                "   - Highlight important details\n"
                "   - Provide context for technical information\n"
                "4. If metadata is missing or outdated:\n"
                "   - Explain what information is unavailable\n"
                "   - Suggest using the fetch agent to update repository data\n"
                "5. For complex queries:\n"
                "   - Break down the information into logical sections\n"
                "   - Explain relationships between different metadata aspects\n"
                "Remember to check metadata fetch time and warn if data might be outdated.",
    tools=[get_repository_metadata],
    before_model_callback=block_keyword_guardrail,
    output_key="last_metadata"
)

# Update Code Analysis agent to remove metadata handling
code_analysis_agent = Agent(
    name="code_analysis_agent_v1",
    model=LiteLlm(model=MODEL_GPT_4O),
    description="Analyzes code using vector similarity search to provide comprehensive code insights.",
    instruction="You are a code analysis assistant that uses vector similarity search to provide detailed explanations. "
                "Your primary goal is to help users understand and analyze code by finding relevant code snippets and providing detailed explanations. "
                "When a user asks a question about the code:\n"
                "1. Use the 'analyze_code_with_vectorstore' tool to search for relevant code snippets\n"
                "2. Analyze the tool's response and provide:\n"
                "   * Summary of found code snippets\n"
                "   * Detailed explanations of how the code works\n"
                "   * Context about where the code fits in the larger codebase\n"
                "   * Patterns and best practices observed\n"
                "3. Consider relevance scores when explaining results:\n"
                "   - Lower scores (closer to 0) indicate better matches\n"
                "   - Explain why certain snippets might be more relevant\n"
                "4. For implementation details:\n"
                "   - Explain the purpose of each code section\n"
                "   - Highlight important functions and classes\n"
                "   - Note any dependencies or requirements\n"
                "5. Suggest follow-up queries for deeper understanding\n"
                "Remember that you need a repository to be fetched first before you can analyze it.\n"
                "For questions about Git history or metadata, suggest using the metadata agent instead.",
    tools=[analyze_code_with_vectorstore],
    before_model_callback=block_keyword_guardrail,
    output_key="last_analysis"
)

# Create Report agent
report_agent = Agent(
    name="report_agent_v1",
    model=LiteLlm(model=MODEL_GPT_4O),
    description="Generates comprehensive reports analyzing code repositories from multiple perspectives.",
    instruction="You are a report generation specialist that creates detailed analyses of code repositories. "
                "Your primary goal is to generate insightful reports that combine code analysis with metadata. "
                "When asked to analyze a repository:\n"
                "1. Plan your analysis approach:\n"
                "   - Define key aspects to analyze (architecture, patterns, etc.)\n"
                "   - Create specific queries for each aspect\n"
                "   - Consider both implementation and metadata\n"
                "2. Generate reports using these query types:\n"
                "   a) Architecture Analysis:\n"
                "      - 'What are the main components and their relationships?'\n"
                "      - 'How is the code organized and structured?'\n"
                "   b) Implementation Details:\n"
                "      - 'How are key features implemented?'\n"
                "      - 'What design patterns are used?'\n"
                "   c) Code Quality:\n"
                "      - 'How is error handling implemented?'\n"
                "      - 'What testing approaches are used?'\n"
                "   d) Development History:\n"
                "      - 'How has the code evolved?'\n"
                "      - 'What are the recent changes?'\n"
                "3. Use the 'generate_analysis_report' tool with your queries\n"
                "4. When presenting results:\n"
                "   - Highlight key findings and patterns\n"
                "   - Provide context for technical details\n"
                "   - Include relevant metadata insights\n"
                "   - Suggest potential improvements\n"
                "5. For specific report requests:\n"
                "   - Focus queries on the requested aspects\n"
                "   - Provide detailed analysis in those areas\n"
                "Remember that a repository must be fetched first before you can analyze it.",
    tools=[generate_analysis_report],
    before_model_callback=block_keyword_guardrail,
    output_key="last_report"
)

# Update Root agent to include report generation
root_agent = Agent(
    name="root_agent_v1",
    model=LiteLlm(model=MODEL_GPT_4O),
    description="Orchestrates GitHub repository fetching, analysis, metadata exploration, and report generation tasks.",
    instruction="You are a code exploration assistant that coordinates repository fetching, analysis, metadata, and reporting tasks. "
                "Your role is to:\n"
                "1. For GitHub repository URLs (containing 'github.com'):\n"
                "   - Delegate to the fetch agent to download and prepare the repository\n"
                "   - The fetch agent will create both a vectorstore and store git metadata\n"
                "2. For code analysis questions (about implementation, patterns, etc.):\n"
                "   - If no repository is fetched yet, guide the user to provide a GitHub URL first\n"
                "   - Otherwise, delegate to the analysis agent to explore the code\n"
                "3. For metadata questions (about commits, branches, tags, history):\n"
                "   - Delegate to the metadata agent for detailed Git information\n"
                "   - The metadata agent can provide commit history, branch details, and more\n"
                "4. For report requests (containing words like 'report', 'analyze', 'summarize'):\n"
                "   - Delegate to the report agent to generate comprehensive analysis\n"
                "   - The report agent will combine code analysis with metadata insights\n"
                "5. For unclear queries:\n"
                "   - Explain the four main operations available:\n"
                "     * Fetching repositories\n"
                "     * Analyzing code implementation\n"
                "     * Exploring Git metadata\n"
                "     * Generating analysis reports\n"
                "   - Guide the user on how to rephrase their request\n"
                "6. Always maintain context between interactions and provide clear guidance.",
    tools=[],  # Root agent doesn't need tools, it delegates to sub-agents
    sub_agents=[github_fetch_agent, code_analysis_agent, git_metadata_agent, report_agent],
    before_model_callback=block_keyword_guardrail,
    output_key="last_delegation"
)
