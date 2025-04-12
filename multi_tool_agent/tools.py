from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from typing import Optional, Dict, Any, Callable, Tuple
import requests
import json
import os
from pathlib import Path
import tempfile
from typing import List, Dict
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitLoader
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import git
import time

# Define constants for paths
DATA_DIR = "multi_tool_agent/data"
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstores")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
CODE_VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "code_vectorstore")

# Ensure directories exist
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def save_repo_metadata(repo_info: Dict, repo_path: str) -> str:
    """
    Saves repository metadata including git information.
    
    Args:
        repo_info (Dict): Repository information from GitHub API
        repo_path (str): Path to the local repository
        
    Returns:
        str: Path to the saved metadata file
    """
    try:
        # Get git repository information
        repo = git.Repo(repo_path)
        
        # Collect git metadata
        git_metadata = {
            "head_commit": str(repo.head.commit),
            "active_branch": repo.active_branch.name,
            "remotes": [{"name": remote.name, "url": remote.url} for remote in repo.remotes],
            "last_commit_info": {
                "message": repo.head.commit.message,
                "author": str(repo.head.commit.author),
                "authored_date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(repo.head.commit.authored_date)),
                "committed_date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(repo.head.commit.committed_date))
            },
            "tags": [str(tag) for tag in repo.tags]
        }
        
        # Combine with GitHub API info
        full_metadata = {
            "github_info": repo_info,
            "git_metadata": git_metadata,
            "vectorstore_path": CODE_VECTORSTORE_PATH,
            "fetch_time": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save metadata
        metadata_file = os.path.join(METADATA_DIR, f"{repo_info['name']}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(full_metadata, f, indent=2)
            
        return metadata_file
        
    except Exception as e:
        print(f"Warning: Could not save complete metadata: {str(e)}")
        # Save basic metadata if git operations fail
        basic_metadata = {
            "github_info": repo_info,
            "vectorstore_path": CODE_VECTORSTORE_PATH,
            "fetch_time": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        metadata_file = os.path.join(METADATA_DIR, f"{repo_info['name']}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(basic_metadata, f, indent=2)
        return metadata_file

class GitHubRepoFetcher(BaseTool):
    """LangChain tool for fetching GitHub repositories."""
    def __init__(self):
        super().__init__(
            name="github_repo_fetcher",
            description="Fetches a GitHub repository and prepares it for analysis by creating a vectorstore.",
            func=self._run
        )
    
    def _run(self, repo_url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        """Fetches a GitHub repository."""
        try:
            # Extract owner and repo name from URL
            parts = repo_url.rstrip('/').split('/')
            owner = parts[-2]
            repo = parts[-1]
            
            # GitHub API endpoints
            api_base = "https://api.github.com"
            repo_api_url = f"{api_base}/repos/{owner}/{repo}"
            clone_url = f"https://github.com/{owner}/{repo}.git"
            
            # Headers for GitHub API
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'GoogleAgents-Fetch'
            }
            
            if 'GITHUB_TOKEN' in os.environ:
                headers['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"
                # Modify clone URL to include token if available
                clone_url = f"https://{os.environ['GITHUB_TOKEN']}@github.com/{owner}/{repo}.git"
            
            # Get repository metadata
            repo_response = requests.get(repo_api_url, headers=headers)
            repo_response.raise_for_status()
            repo_info = repo_response.json()
            
            # Create temporary directory for the repository
            temp_dir = tempfile.mkdtemp()
            repo_dir = os.path.join(temp_dir, repo)
            
            # Clone the repository
            print(f"--- Tool: Cloning repository to {repo_dir} ---")
            git.Repo.clone_from(clone_url, repo_dir)
            
            # Load and split text into chunks
            documents, splits = self.load_and_split_repo(repo_dir)

            # Create a vectorstore from the splits
            vectorstore = self.create_code_vectorstore(splits, persist_directory=CODE_VECTORSTORE_PATH)
            
            # Store basic repo info
            repo_info = {
                "name": repo_info["name"],
                "description": repo_info["description"],
                "url": repo_info["html_url"],
                "local_path": repo_dir,
                "vectorstore_path": CODE_VECTORSTORE_PATH
            }
            
            # Save extended metadata
            metadata_file = save_repo_metadata(repo_info, repo_dir)
            repo_info["metadata_file"] = metadata_file
            
            return {
                "status": "success",
                "repo_info": repo_info,
                "message": f"Successfully fetched repository {owner}/{repo} and created vectorstore"
            }
            
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error_message": f"Error fetching repository: {str(e)}"}
        except git.exc.GitCommandError as e:
            return {"status": "error", "error_message": f"Error cloning repository: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error_message": f"Error processing repository: {str(e)}"}
        finally:
            # Clean up temporary directory
            if 'temp_dir' in locals():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up temporary directory: {str(e)}")

    async def _arun(self, repo_url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        """Async implementation - we just call the sync version for now."""
        return self._run(repo_url, run_manager)
    
    def load_and_split_repo(self, repo_path: str) -> Tuple[List[Document], List[Document]]:
        """
        Loads a local GitHub repository using LangChain's GitLoader and splits the content
        using RecursiveCharacterTextSplitter.
        
        Args:
            repo_path (str): Path to the local repository
            
        Returns:
            Tuple[List[Document], List[Document]]: A tuple containing:
                - The original documents from GitLoader
                - The split documents from RecursiveCharacterTextSplitter
        """
        try:
            # Initialize GitLoader with the repository path
            loader = GitLoader(
                repo_path=repo_path,
                branch=None  # Let GitLoader detect the current branch
            )
            
            # Load all documents from the repository
            documents = loader.load()
            
            # Initialize the text splitter with specified parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            
            # Split the documents
            splits = text_splitter.split_documents(documents)
            
            return documents, splits
                
        except Exception as e:
            print(f"Error processing repository: {str(e)}")
            raise

    def create_code_vectorstore(self, splits: List[Document], persist_directory: Optional[str] = None) -> FAISS:
        """
        Creates a FAISS vectorstore from document splits using OpenAI embeddings.
        
        Args:
            splits (List[Document]): List of document splits to embed
            persist_directory (Optional[str]): Directory to persist the vectorstore to. If None, store in memory only.
            
        Returns:
            FAISS: The created vectorstore containing the embedded documents
        """
        try:
            # Initialize OpenAI embeddings
            embeddings = OpenAIEmbeddings()
            
            # Create the FAISS vectorstore from the documents
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # If a persist directory is provided, save the vectorstore
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                vectorstore.save_local(persist_directory)
                print(f"Vectorstore saved to {persist_directory}")
            
            return vectorstore
            
        except Exception as e:
            print(f"Error creating vectorstore: {str(e)}")
            raise

def fetch_github_repo(repo_url: str, tool_context: ToolContext) -> dict:
    """Tool wrapper for GitHubRepoFetcher that handles state management."""
    print(f"--- Tool: fetch_github_repo called for URL: {repo_url} ---")
    
    fetcher = GitHubRepoFetcher()
    result = fetcher._run(repo_url)
    
    if result["status"] == "success" and isinstance(result.get("repo_info"), dict):
        tool_context.state["last_fetched_repo"] = result["repo_info"]
    
    return result

def load_repo_metadata(repo_name: str) -> Optional[Dict]:
    """
    Loads repository metadata from the metadata directory.
    
    Args:
        repo_name (str): Name of the repository
        
    Returns:
        Optional[Dict]: Repository metadata if found, None otherwise
    """
    try:
        metadata_file = os.path.join(METADATA_DIR, f"{repo_name}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Warning: Could not load metadata: {str(e)}")
        return None

class CodeAnalyzer(BaseTool):
    """LangChain tool for analyzing code using vector similarity search."""
    def __init__(self):
        super().__init__(
            name="code_analyzer",
            description="Analyzes code using vector similarity search and repository metadata.",
            func=self._run
        )
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        """Analyzes code using vectorstore similarity search."""
        try:
            embeddings = OpenAIEmbeddings()
            
            if not os.path.exists(CODE_VECTORSTORE_PATH):
                return {
                    "status": "error",
                    "error_message": "Vector store not found. Please fetch a repository first."
                }
                
            # Load the vectorstore with allow_dangerous_deserialization=True
            vectorstore = FAISS.load_local(
                folder_path=CODE_VECTORSTORE_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            
            results = vectorstore.similarity_search_with_score(query, k=5)
            
            formatted_results = []
            repo_name = None
            
            for doc, score in results:
                # Try to extract repo name from metadata or file path
                if not repo_name:
                    if 'repo_name' in doc.metadata:
                        repo_name = doc.metadata['repo_name']
                    elif 'source' in doc.metadata:
                        # Extract repo name from file path
                        path_parts = doc.metadata['source'].split(os.path.sep)
                        if len(path_parts) > 1:
                            repo_name = path_parts[-2]  # Assuming repo name is parent directory
                
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                })
            
            response = {
                "status": "success",
                "query": query,
                "results": formatted_results
            }
            
            # Try to load and include repository metadata if we found a repo name
            if repo_name:
                repo_metadata = load_repo_metadata(repo_name)
                if repo_metadata:
                    response["repository_metadata"] = repo_metadata
                    # Extract and highlight git-specific information
                    if "git_metadata" in repo_metadata:
                        response["git_info"] = {
                            "last_commit": repo_metadata["git_metadata"].get("last_commit_info", {}),
                            "active_branch": repo_metadata["git_metadata"].get("active_branch", "unknown"),
                            "tags": repo_metadata["git_metadata"].get("tags", [])
                        }
            
            return response
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error analyzing code: {str(e)}"
            }

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        """Async implementation - we just call the sync version for now."""
        return self._run(query, run_manager)

def analyze_code_with_vectorstore(query: str, tool_context: ToolContext) -> dict:
    """Tool wrapper for CodeAnalyzer that handles state management."""
    analyzer = CodeAnalyzer()
    result = analyzer._run(query)
    
    if result["status"] == "success":
        # Add repository context from state if available
        last_repo = tool_context.state.get("last_fetched_repo", {})
        if isinstance(last_repo, dict):
            result["repository"] = {
                "name": last_repo.get("name", "Unknown"),
                "url": last_repo.get("url", "Unknown")
            }
            # Try to load metadata for the repository
            if "name" in last_repo:
                repo_metadata = load_repo_metadata(last_repo["name"])
                if repo_metadata:
                    result["repository_metadata"] = repo_metadata
                    # Extract and highlight git-specific information
                    if "git_metadata" in repo_metadata:
                        result["git_info"] = {
                            "last_commit": repo_metadata["git_metadata"].get("last_commit_info", {}),
                            "active_branch": repo_metadata["git_metadata"].get("active_branch", "unknown"),
                            "tags": repo_metadata["git_metadata"].get("tags", [])
                        }
        else:
            result["repository"] = {
                "name": "Unknown",
                "url": "Unknown"
            }
        tool_context.state["last_analysis"] = result
    
    return result

class GitMetadataRetriever(BaseTool):
    """LangChain tool for retrieving Git metadata from repositories."""
    def __init__(self):
        super().__init__(
            name="git_metadata_retriever",
            description="Retrieves Git metadata including commit history, branches, and tags from a repository.",
            func=self._run
        )
    
    def _run(self, repo_name: str = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        """Retrieves Git metadata for a repository."""
        try:
            # If no repo name provided, try to find the most recent one
            if not repo_name:
                # List all metadata files and get the most recent one
                metadata_files = [f for f in os.listdir(METADATA_DIR) if f.endswith('_metadata.json')]
                if not metadata_files:
                    return {
                        "status": "error",
                        "error_message": "No repository metadata found. Please fetch a repository first."
                    }
                # Sort by modification time and get the most recent
                latest_file = max(
                    [os.path.join(METADATA_DIR, f) for f in metadata_files],
                    key=os.path.getmtime
                )
                repo_name = latest_file.split('/')[-1].replace('_metadata.json', '')
            else:
                # Clean up repo name if it's a full GitHub URL or path
                repo_name = repo_name.rstrip('/')
                if '/' in repo_name:
                    # Extract just the repo name from owner/repo format
                    repo_name = repo_name.split('/')[-1]
            
            # Try different metadata file possibilities
            possible_filenames = [
                f"{repo_name}_metadata.json",
                f"GoogleAgents_metadata.json"  # Fallback for known repo
            ]
            
            metadata = None
            for filename in possible_filenames:
                try:
                    metadata_path = os.path.join(METADATA_DIR, filename)
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            break
                except Exception as e:
                    print(f"Warning: Could not load metadata from {filename}: {str(e)}")
                    continue
            
            if not metadata:
                return {
                    "status": "error",
                    "error_message": f"No metadata found for repository: {repo_name}"
                }
            
            # Extract relevant git information
            git_info = metadata.get("git_metadata", {})
            
            return {
                "status": "success",
                "repository_name": repo_name,
                "git_metadata": {
                    "current_commit": {
                        "hash": git_info.get("head_commit", "unknown"),
                        "message": git_info.get("last_commit_info", {}).get("message", "unknown"),
                        "author": git_info.get("last_commit_info", {}).get("author", "unknown"),
                        "authored_date": git_info.get("last_commit_info", {}).get("authored_date", "unknown"),
                        "committed_date": git_info.get("last_commit_info", {}).get("committed_date", "unknown")
                    },
                    "branch_info": {
                        "active_branch": git_info.get("active_branch", "unknown"),
                        "remotes": git_info.get("remotes", [])
                    },
                    "tags": git_info.get("tags", []),
                    "repository_info": metadata.get("github_info", {}),
                    "fetch_time": metadata.get("fetch_time", "unknown")
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error retrieving git metadata: {str(e)}"
            }

    async def _arun(self, repo_name: str = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        """Async implementation - we just call the sync version for now."""
        return self._run(repo_name, run_manager)

def get_repository_metadata(repo_name: Optional[str], tool_context: ToolContext) -> dict:
    """Tool wrapper for GitMetadataRetriever that handles state management."""
    print(f"--- Tool: get_repository_metadata called for repo: {repo_name or 'most recent'} ---")
    
    # Try to get repo name from state if not provided
    if not repo_name and isinstance(tool_context.state.get("last_fetched_repo"), dict):
        repo_name = tool_context.state["last_fetched_repo"].get("name")
    
    retriever = GitMetadataRetriever()
    result = retriever._run(repo_name)
    
    if result["status"] == "success":
        # Store the retrieved metadata in the tool context
        tool_context.state["last_retrieved_metadata"] = result["git_metadata"]
    
    return result

class ReportGenerator(BaseTool):
    """LangChain tool for generating comprehensive code analysis reports."""
    def __init__(self):
        super().__init__(
            name="report_generator",
            description="Generates detailed reports combining code analysis, metadata, and repository insights.",
            func=self._run
        )
    
    def _run(
        self, 
        analysis_queries: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict:
        """Generates a comprehensive report by analyzing code with multiple queries."""
        try:
            # Initialize analyzers
            code_analyzer = CodeAnalyzer()
            metadata_retriever = GitMetadataRetriever()
            
            report_sections = []
            metadata_overview = None
            
            # First, get repository metadata
            metadata_result = metadata_retriever._run()
            if metadata_result["status"] == "success":
                metadata_overview = {
                    "repository": metadata_result["repository_name"],
                    "last_commit": metadata_result["git_metadata"]["current_commit"],
                    "branch": metadata_result["git_metadata"]["branch_info"]["active_branch"],
                    "fetch_time": metadata_result["git_metadata"]["fetch_time"]
                }
            
            # Run each analysis query
            for query in analysis_queries:
                analysis_result = code_analyzer._run(query)
                if analysis_result["status"] == "success":
                    section = {
                        "query": query,
                        "findings": [],
                        "relevance_summary": []
                    }
                    
                    # Process results
                    for result in analysis_result.get("results", []):
                        section["findings"].append({
                            "content": result["content"],
                            "file": result["metadata"].get("source", "Unknown"),
                            "relevance_score": result["relevance_score"]
                        })
                        
                        # Add relevance summary for top matches
                        if result["relevance_score"] < 1.0:  # Lower is better
                            section["relevance_summary"].append(
                                f"Found relevant code in {result['metadata'].get('source', 'Unknown')} "
                                f"(score: {result['relevance_score']:.3f})"
                            )
                    
                    report_sections.append(section)
            
            # Compile the final report
            report = {
                "status": "success",
                "metadata": metadata_overview,
                "sections": report_sections,
                "summary": {
                    "total_queries": len(analysis_queries),
                    "successful_queries": len([s for s in report_sections if s["findings"]]),
                    "total_findings": sum(len(s["findings"]) for s in report_sections),
                    "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Save the report
            os.makedirs(os.path.join(DATA_DIR, "reports"), exist_ok=True)
            report_file = os.path.join(
                DATA_DIR, 
                "reports", 
                f"analysis_report_{int(time.time())}.json"
            )
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            report["report_file"] = report_file
            return report
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error generating report: {str(e)}"
            }

    async def _arun(
        self,
        analysis_queries: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict:
        """Async implementation - we just call the sync version for now."""
        return self._run(analysis_queries, run_manager)

def generate_analysis_report(analysis_queries: List[str], tool_context: ToolContext) -> dict:
    """Tool wrapper for ReportGenerator that handles state management."""
    print(f"--- Tool: generate_analysis_report called with {len(analysis_queries)} queries ---")
    
    generator = ReportGenerator()
    result = generator._run(analysis_queries)
    
    if result["status"] == "success":
        # Store the generated report in the tool context
        tool_context.state["last_generated_report"] = result
        
        # Also store individual section results for future reference
        tool_context.state["report_sections"] = {
            section["query"]: section["findings"]
            for section in result["sections"]
        }
    
    return result


