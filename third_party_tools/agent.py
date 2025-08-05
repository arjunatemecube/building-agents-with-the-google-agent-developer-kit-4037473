from google.adk.agents import LlmAgent
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from google.adk.tools.langchain_tool import LangchainTool

# Initialize the Wikipedia tool from LangChain
wikipedia_api_wrapper = WikipediaAPIWrapper(
    top_k_results=1, 
    doc_content_chars_max=500
)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

adk_wikipedia_tool = LangchainTool(tool=wikipedia_tool)

wikipedia_summarizer_agent = LlmAgent(
    name="WikipediaSummarizerAgent",
    model="gemini-2.0-flash",
    instruction=(
        "You are an academic summarizer. "
        "Use the Wikipedia tool to find information on the given topic and then provide a concise summary of the key points. "
        "Focus on factual information and avoid speculation."
    ),
    description="Fetches and summarizes information from Wikipedia using a third-party LangChain tool.",
    tools=[adk_wikipedia_tool],
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2
    )
)

root_agent = wikipedia_summarizer_agent