import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import json

# Load environment variables
load_dotenv()

# Load data
DATA_PATH = "data/cleaned_game_data.csv"
df = pd.read_csv(DATA_PATH)

# Initialize LLM
llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

PREFIX = """You are a game data analyst. You work with a pandas dataframe 'df'.
The dataframe has columns: name, steam_appid, is_free, supported_languages, developers, publishers, categories, genres, release_date, positive, negative, price_usd, tags, total_ratings, rating, price_inr, release_year.

You MUST follow this format strictly:
Thought: your reasoning
Action: python_repl_ast
Action Input: your python code
Observation: result
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I have the final answer. I will now provide the result as JSON.
Final Answer: {{"response": "your human-readable answer", "metadata": {{"key": "value"}}}}

IMPORTANT:
1. Do NOT include any 'Action' or 'Action Input' after you write 'Final Answer'.
2. The 'metadata' field should containing specific game details (like appid, rating, price) if relevant to the query.
"""

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    prefix=PREFIX,
    handle_parsing_errors="Check your output format and ensure you provide a single Final Answer as JSON."
)

test_queries = [
    "Do free games have higher ratings on average than paid games?",
    "How many Action games support Korean?",
    "Best multiplayer shooters released after 2015",
    "What is the price of Counter Strike game in INR and USD?"
]

print("Starting Verification...")

for query in test_queries:
    print(f"\nQuery: {query}")
    try:
        result = agent.invoke({"input": query})
        print(f"Result: {result.get('output')}")
    except Exception as e:
        print(f"Error: {e}")

print("\nVerification Complete.")
