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
    prefix = """You are a data analysis assistant. You have a pandas dataframe called 'df'.
RULES:
1. Use the python_repl_ast tool to run Python code on df.
2. Get the answer efficiently.
3. Once you have the answer, respond with Final Answer immediately.
4. For charts: save to './chart_image/chart.png' and reply: {{"chart": "Chart generated successfully"}}
5. For tables: reply with: {{"table": {{"columns": [...], "data": [...]}}}}
6. For text answers: reply with: {{"answer": "your answer here"}}
"""

    # Create the agent
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        prefix=prefix,
        include_df_in_prompt=True,
        number_of_head_rows=3,
        max_iterations=8,
        early_stopping_method="generate"
    )

def query_pd_agent(agent, query):
    """
    Query the agent and return the response.
    """
    # We now pass the raw query because the rules are in the prefix
    try:
        # We add a small reminder to the query to ensure JSON output
        full_query = f"{query}. REMEMBER: Your final answer MUST be a valid JSON string as per the rules."
        response = agent.run(full_query)
        return str(response)
    except Exception as e:
        error_msg = str(e)
        if "Could not parse LLM output" in error_msg:
            return str({"answer": "I processed the data but had trouble formatting the response. Please check if your chart or table appeared below."})
        return str({"answer": f"Error: {error_msg}"})
