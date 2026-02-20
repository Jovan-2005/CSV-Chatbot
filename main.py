from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import json
import ast
# Load environment variable
load_dotenv()

app = FastAPI(title="Game Analytics Agent")

# Load data
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned_game_data.csv")
df = pd.read_csv(DATA_PATH)

# Initialize LLM
llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Custom Prompt to ensure JSON output with metadata
PREFIX = """You are a game data analyst. You work with a pandas dataframe 'df'.
The dataframe has columns: name, steam_appid, is_free, supported_languages, developers, publishers, categories, genres, release_date, positive, negative, price_usd, tags, total_ratings, rating, price_inr, release_year.

You MUST follow this format strictly:
Thought: your reasoning
Action: python_repl_ast
Action Input: your python code
Observation: result
... (repeat Thought/Action/Action Input/Observation if needed)

When you are ready to provide the final answer, do NOT use the 'python_repl_ast' tool. Instead, use exactly this format:
Thought: I have the final answer.
Final Answer: {{"response": "your human-readable answer", "metadata": {{"key": "value"}}}}

IMPORTANT:
1. The 'Final Answer' must be a valid JSON object.
2. The 'metadata' field should contain specific game details (like appid, rating, price) if relevant.
3. NEVER call an action after providing the Final Answer.
"""

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    prefix=PREFIX,
    handle_parsing_errors="Check your output format and ensure you provide a single Final Answer as JSON."
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    metadata: dict

@app.post("/game/analytics", response_model=QueryResponse)
async def get_game_analytics(request: QueryRequest):
    try:
        result = agent.invoke({"input": request.query})
        output = result.get("output", "")
        
        # Try to parse the output as JSON
        try:
            # Clean up potential markdown code blocks
            clean_output = output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:-3].strip()
            elif clean_output.startswith("```"):
                clean_output = clean_output[3:-3].strip()
            
            data = json.loads(clean_output)
            return QueryResponse(
                response=data.get("response", output),
                metadata=data.get("metadata", {})
            )
        except:
            # Fallback if parsing fails but it's a string
            return QueryResponse(
                response=output,
                metadata={}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
