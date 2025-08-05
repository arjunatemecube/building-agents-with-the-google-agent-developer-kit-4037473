from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.genai import types

root_agent = LlmAgent(
    name="FactFinderAgent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a knowledgeable fact-finding assistant. "
        "Use the Google Search tool to find answers to factual questions. "
        "Summarize the information concisely and provide the answer."
    ),
    description="Answers questions factually by performing a Google search",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3
    ),
    tools=[google_search]
)