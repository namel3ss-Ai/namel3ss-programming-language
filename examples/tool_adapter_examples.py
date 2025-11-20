"""
Example usage of Tool Adapters with all N3 LLM providers.

Demonstrates OpenAPI, LangChain, and LLM tool creation with
OpenAI, Anthropic, Vertex AI, Azure OpenAI, and Ollama.
"""

import asyncio
from n3_server.adapters import (
    OpenAPIAdapter,
    LangChainAdapter,
    LLMToolWrapper,
    create_llm_tool,
)
from n3_server.adapters.llm_tool_wrapper import (
    create_openai_tool,
    create_anthropic_tool,
    create_vertex_tool,
    create_azure_tool,
    create_ollama_tool,
)


async def example_openapi_import():
    """Example: Import tools from OpenAPI specification."""
    print("\n=== OpenAPI Import Example ===")
    
    adapter = OpenAPIAdapter()
    
    # Example: Import GitHub API tools
    tools = await adapter.import_from_url(
        spec_url="https://api.github.com/openapi.json",
        base_url="https://api.github.com",
        name_prefix="github_",
        operation_filter=lambda op: op.method == "GET" and "repos" in op.path,
    )
    
    print(f"Imported {len(tools)} GitHub API tools:")
    for tool in tools:
        metadata = tool._tool_metadata
        print(f"  - {metadata['name']}: {metadata['description']}")
    
    await adapter.close()


async def example_langchain_import():
    """Example: Import LangChain tools."""
    print("\n=== LangChain Import Example ===")
    
    try:
        from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
        
        adapter = LangChainAdapter()
        
        # Import search tools
        search = DuckDuckGoSearchRun()
        search_tool = adapter.import_tool(search, name_prefix="web_")
        
        wiki = WikipediaQueryRun()
        wiki_tool = adapter.import_tool(wiki, name_prefix="wiki_")
        
        print("Imported LangChain tools:")
        print(f"  - {search_tool.__name__}: {search_tool.__doc__}")
        print(f"  - {wiki_tool.__name__}: {wiki_tool.__doc__}")
        
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain duckduckgo-search wikipedia")


async def example_openai_tools():
    """Example: Create OpenAI-powered tools."""
    print("\n=== OpenAI LLM Tools ===")
    
    # Summarizer
    summarizer = create_openai_tool(
        name="gpt_summarize",
        description="Summarize text using GPT-4",
        model="gpt-4",
        system_prompt="You are an expert at creating concise summaries.",
        temperature=0.5,
        max_tokens=300,
    )
    print(f"Created: {summarizer.__name__}")
    
    # Translator
    translator = create_openai_tool(
        name="gpt_translate",
        description="Translate text using GPT-4",
        model="gpt-4",
        system_prompt="You are a professional translator.",
        temperature=0.3,
    )
    print(f"Created: {translator.__name__}")
    
    # Code reviewer
    reviewer = create_openai_tool(
        name="gpt_code_review",
        description="Review code using GPT-4",
        model="gpt-4",
        system_prompt="You are an expert code reviewer. Identify bugs, security issues, and suggest improvements.",
        temperature=0.2,
    )
    print(f"Created: {reviewer.__name__}")


async def example_anthropic_tools():
    """Example: Create Anthropic (Claude) powered tools."""
    print("\n=== Anthropic Claude Tools ===")
    
    # Text analyzer
    analyzer = create_anthropic_tool(
        name="claude_analyze",
        description="Analyze text deeply",
        model="claude-3-sonnet-20240229",
        system_prompt="You are an expert at deep text analysis.",
        temperature=0.7,
    )
    print(f"Created: {analyzer.__name__}")
    
    # Creative writer
    writer = create_anthropic_tool(
        name="claude_write",
        description="Write creative content",
        model="claude-3-opus-20240229",
        system_prompt="You are a creative writer specializing in engaging content.",
        temperature=0.9,
    )
    print(f"Created: {writer.__name__}")


async def example_vertex_tools():
    """Example: Create Vertex AI (Gemini) tools."""
    print("\n=== Vertex AI Gemini Tools ===")
    
    # General assistant
    assistant = create_vertex_tool(
        name="gemini_assist",
        description="General purpose assistant",
        model="gemini-pro",
        system_prompt="You are a helpful AI assistant.",
        temperature=0.7,
    )
    print(f"Created: {assistant.__name__}")
    
    # Data analyzer
    data_analyzer = create_vertex_tool(
        name="gemini_data",
        description="Analyze data patterns",
        model="gemini-pro",
        system_prompt="You are a data analysis expert.",
        temperature=0.4,
    )
    print(f"Created: {data_analyzer.__name__}")


async def example_azure_tools():
    """Example: Create Azure OpenAI tools."""
    print("\n=== Azure OpenAI Tools ===")
    
    # Customer support
    support = create_azure_tool(
        name="azure_support",
        description="Customer support assistant",
        deployment_name="gpt-4-deployment",
        system_prompt="You are a friendly customer support agent.",
        temperature=0.6,
    )
    print(f"Created: {support.__name__}")
    
    # Technical documenter
    documenter = create_azure_tool(
        name="azure_docs",
        description="Generate technical documentation",
        deployment_name="gpt-4-deployment",
        system_prompt="You are a technical writer creating clear documentation.",
        temperature=0.3,
    )
    print(f"Created: {documenter.__name__}")


async def example_ollama_tools():
    """Example: Create Ollama (local) tools."""
    print("\n=== Ollama Local Model Tools ===")
    
    # Local chat
    chat = create_ollama_tool(
        name="local_chat",
        description="Chat with local LLM",
        model="llama2",
        system_prompt="You are a helpful assistant running locally.",
        temperature=0.7,
    )
    print(f"Created: {chat.__name__}")
    
    # Local coder
    coder = create_ollama_tool(
        name="local_code",
        description="Code generation with local model",
        model="codellama",
        system_prompt="You are an expert programmer.",
        temperature=0.2,
    )
    print(f"Created: {coder.__name__}")


async def example_structured_output():
    """Example: LLM tool with structured JSON output."""
    print("\n=== Structured Output Example ===")
    
    wrapper = LLMToolWrapper()
    
    # Entity extractor with JSON output
    extractor = wrapper.create_tool(
        name="extract_entities",
        description="Extract named entities from text",
        llm_name="gpt4",
        response_format="json",
        output_schema={
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of people mentioned"
                },
                "organizations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Organizations mentioned"
                },
                "locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Locations mentioned"
                },
                "dates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Dates mentioned"
                }
            }
        },
        system_prompt="Extract people, organizations, locations, and dates from the text. Return as JSON.",
    )
    
    print(f"Created structured tool: {extractor.__name__}")
    print("Output schema:", extractor._tool_metadata["output_schema"])


async def example_multi_llm_workflow():
    """Example: Workflow using multiple LLM providers."""
    print("\n=== Multi-LLM Workflow Example ===")
    
    wrapper = LLMToolWrapper()
    
    # Step 1: GPT-4 for initial analysis
    analyzer = wrapper.create_tool(
        name="analyze_with_gpt",
        description="Deep analysis with GPT-4",
        llm_name="gpt4",
        system_prompt="Analyze the text thoroughly.",
        temperature=0.5,
    )
    
    # Step 2: Claude for creative expansion
    expander = wrapper.create_tool(
        name="expand_with_claude",
        description="Creative expansion with Claude",
        llm_name="claude3",
        system_prompt="Expand on the analysis creatively.",
        temperature=0.8,
    )
    
    # Step 3: Gemini for final summary
    summarizer = wrapper.create_tool(
        name="summarize_with_gemini",
        description="Final summary with Gemini",
        llm_name="gemini",
        system_prompt="Create a concise final summary.",
        temperature=0.3,
    )
    
    print("Created multi-LLM workflow:")
    print(f"  1. {analyzer.__name__} (OpenAI GPT-4)")
    print(f"  2. {expander.__name__} (Anthropic Claude)")
    print(f"  3. {summarizer.__name__} (Vertex Gemini)")


async def example_tool_registration():
    """Example: Register tools in ToolRegistry."""
    print("\n=== Tool Registration Example ===")
    
    from n3_server.api.tools import registry
    
    # Create various tools
    summarizer = create_openai_tool("summarize", "Summarize text", model="gpt-4")
    translator = create_anthropic_tool("translate", "Translate text", model="claude-3-sonnet-20240229")
    analyzer = create_vertex_tool("analyze", "Analyze content", model="gemini-pro")
    
    # Register tools
    tools = [summarizer, translator, analyzer]
    for tool in tools:
        metadata = tool._tool_metadata
        registry.register(
            tool,
            description=metadata["description"],
            tags=metadata["tags"],
        )
    
    print(f"Registered {len(tools)} tools in registry")
    print("Available tools:", [t.name for t in registry.list_tools()])


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Tool Adapters - Complete Examples")
    print("=" * 60)
    
    # OpenAPI examples
    await example_openapi_import()
    
    # LangChain examples
    await example_langchain_import()
    
    # LLM tool examples for each provider
    await example_openai_tools()
    await example_anthropic_tools()
    await example_vertex_tools()
    await example_azure_tools()
    await example_ollama_tools()
    
    # Advanced examples
    await example_structured_output()
    await example_multi_llm_workflow()
    await example_tool_registration()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
