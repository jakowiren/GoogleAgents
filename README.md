# GoogleAgents

A multi-agent system for GitHub repository analysis built with Google's Agent Development Kit (ADK). The system combines code analysis, Git metadata exploration, and comprehensive reporting capabilities.

## Features

- 🔍 **Repository Analysis**: Fetch and analyze GitHub repositories
- 📊 **Code Insights**: Vector-based code search and pattern recognition
- 🌳 **Git Metadata**: Explore commit history, branches, and development patterns
- 📝 **Report Generation**: Create comprehensive code analysis reports

## Tools Used

- **Google ADK**: Core agent framework
- **LangChain**: Document processing and vector operations
- **FAISS**: Vector similarity search
- **OpenAI Embeddings**: Code embedding generation
- **GitPython**: Git repository handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jakowiren/GoogleAgents.git
cd GoogleAgents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```bash
OPENAI_API_KEY=your_openai_key
GITHUB_TOKEN=your_github_token  # Optional for public repos
```

## Usage

### Command Line Interface

Run the test workflow:
```bash
python -m multi_tool_agent.main
```

### Web ADK Interface

1. Start the Web ADK server:
```bash
adk web
```

2. Access the interface at `http://localhost:8080`

3. Available agents:
   - **Root Agent**: Orchestrates all operations
   - **Fetch Agent**: Downloads and prepares repositories
   - **Analysis Agent**: Performs code analysis
   - **Metadata Agent**: Explores Git history
   - **Report Agent**: Generates comprehensive reports

## Example Workflow

1. Fetch a repository:
```
Can you fetch this repository: https://github.com/username/repo
```

2. Analyze code:
```
How is the code organized in this repository?
```

3. Generate a report:
```
Generate a comprehensive report analyzing the code architecture
```

## License

MIT

## Acknowledgments

Built with [Google's Agent Development Kit](https://google.github.io/adk-docs/)