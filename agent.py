# Import necessary libraries.
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def create_pd_agent(filename: str, api_key: str):
    """
    Create a Pandas DataFrame agent from a CSV file using Groq (Llama 3).
    """
    # Set the API key
    if not api_key:
        raise ValueError("Groq API Key is required.")
        
    os.environ["GROQ_API_KEY"] = api_key
    
    # Read the CSV file into a Pandas DataFrame.
    try:
        df = pd.read_csv(filename)
    except UnicodeDecodeError:
        df = pd.read_csv(filename, encoding='latin1')
    
    # Initialize the Groq model
    model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_retries=5,
        request_timeout=60
    )

    # Custom prompt to force tool usage and JSON output
    prefix = """You are a helpful data analyst. You are working with a pandas dataframe 'df'.
You MUST use the following format for every step:

Thought: you should always think about what to do
Action: the action to take, should be exactly 'python_repl_ast'
Action Input: the python code to execute
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: your final response (MUST be JSON as per rules below)

RULES FOR FINAL ANSWER:
1. If you output a table, format it as a JSON object:
{{"table": {{"columns": ["col1", "col2"], "data": [[1, 2], [3, 4]]}}}}
2. If you output text, wrap it in:
{{"answer": "your text"}}

Begin!
"""

    # Create the agent
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        prefix=prefix,
        include_df_in_prompt=True,
        number_of_head_rows=5,
        max_iterations=20,
        early_stopping_method="force"
    )

def query_pd_agent(agent, query):
    """
    Query the agent and return the response.
    """
    # We now pass the raw query because the rules are in the prefix
    try:
        # We add a small reminder to the query to ensure JSON output
        full_query = f"{query}. IMPORTANT: Please try to answer using the JSON format (answer or table keys) if possible."
        response = agent.run(full_query)
        return str(response)
    except Exception as e:
        error_msg = str(e)
        if "Could not parse LLM output" in error_msg:
            return str({"answer": "I processed the data but had trouble formatting the response. Please check if your chart or table appeared below."})
        return str({"answer": f"Error: {error_msg}"})
