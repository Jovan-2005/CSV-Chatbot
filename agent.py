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
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_retries=5,
        request_timeout=60
    )

    # Custom prompt to force tool usage and JSON output
    prefix = """
    You are a data analysis expert. You have access to a pandas dataframe 'df'.
    
    IMPORTANT RULES:
    1. If you need to answer a question about the data, ALWAYS use the 'python_repl_ast' tool.
    2. NEVER return python code as your Final Answer. You must EXECUTE it.
    3. For charts: If asked for a visualization, EXECUTE code to create the plot, save it to './chart_image/chart.png', and then your Final Answer must be EXACTLY: {{"chart": "Chart generated successfully"}}
    4. For tables: Your Final Answer must be a JSON object like: {{"table": {{"columns": ["col1", "col2"], "data": [[1, 2], [3, 4]]}}}}
    5. For text: Your Final Answer must be a JSON object like: {{"answer": "Your detailed summary"}}
    """

    # Create the agent
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prefix=prefix,
        include_df_in_prompt=True,
        number_of_head_rows=5
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
