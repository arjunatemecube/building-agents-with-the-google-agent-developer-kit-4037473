from google.adk.agents import LlmAgent, SequentialAgent

GEMINI_MODEL = "gemini-2.0-flash"

idea_generator_agent = LlmAgent(
    name="IdeaGeneratorAgent",
    model=GEMINI_MODEL,
    instruction="""You are a creative content idea generator.
Based on the user's provided topic, brainstorm and list 3-5 unique and engaging content ideas.
Output *only* a numbered list of ideas. Do not add any other text.
Example: '
- 10 Ways to Master Python 
- The Future of AI in Daily Life 
- Building Your First Web App with Flask
'
""",
    description="Generates initial content ideas for a given topic.",
    output_key="content_ideas" 
)

keyword_research_agent = LlmAgent(
    name="KeywordResearchAgent",
    model=GEMINI_MODEL,
    instruction="""You are an SEO keyword research expert.
Given a list of content ideas, identify 3-5 primary and secondary keywords for each idea that a target audience would search for.

**Content Ideas:**
{content_ideas}

**Output Format:**
For each idea, list relevant keywords.
Example:
Idea: The Future of AI in Daily Life
Keywords: AI in daily life, future AI, AI impact, personal AI, everyday artificial intelligence
---
Idea: Building Your First Web App with Flask
Keywords: Flask web app, Flask tutorial, build web app python, Python web development, Flask for beginners
""",
    description="Identifies relevant SEO keywords for content ideas.",
    output_key="seo_keywords_map", # Stores output in state['seo_keywords_map']
)

content_creation_pipeline = SequentialAgent(
    name="ContentCreationPipeline",
    sub_agents=[
        idea_generator_agent,
        keyword_research_agent
    ],
    description="Executes a sequence for content idea generation, keyword research, and SEO outline creation."
)

root_agent = content_creation_pipeline