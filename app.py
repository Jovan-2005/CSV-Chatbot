import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from agent import create_pd_agent

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object."""
    try:
        # Clean up the response string, sometimes LLMs add markdown code blocks
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        return json.loads(clean_response)
    except json.JSONDecodeError:
        return {"answer": response}

def write_response(response_dict: dict):
    """
    Write a response from an agent to the Streamlit app.
    
    Args:
        response_dict: The response from the agent.
    """
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    if "chart" in response_dict:
        st.image("./chart_image/chart.png")
    
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

# Page configuration
st.set_page_config(page_title="CSV Chat Expert", layout="wide")

st.title("🤖 Intelligent CSV Chat & Analysis")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Load key strictly from environment
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("⚠️ API Key missing! Please add GROQ_API_KEY to your .env file.")
        st.stop()
    
    st.success("✅ API Key Loaded Securely")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if st.button("Clear Chat History"):

        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        elif message["type"] == "image":
            st.image(message["content"])
        elif message["type"] == "table":
            st.table(message["content"])

# Main chat logic
if api_key and uploaded_file:
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("chart_image", exist_ok=True)
    
    # Save uploaded file to data directory
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Cache agent in session_state so it's not recreated on every rerun
    if "agent" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        try:
            with st.spinner("Initializing AI agent..."):
                st.session_state.agent = create_pd_agent(file_path, api_key)
                st.session_state.current_file = uploaded_file.name
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()
    
    agent = st.session_state.agent
    
    # User input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    result = agent.invoke({"input": prompt})
                    response_str = result.get("output", str(result)) if isinstance(result, dict) else str(result)
                    decoded_response = decode_response(response_str)
                    
                    # Handle different response types
                    if "answer" in decoded_response:
                        st.write(decoded_response["answer"])
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": decoded_response["answer"]})
                    
                    if "chart" in decoded_response:
                        st.image("./chart_image/chart.png")
                        st.session_state.messages.append({"role": "assistant", "type": "image", "content": "./chart_image/chart.png"})
                        
                    if "table" in decoded_response:
                        data = decoded_response["table"]
                        df = pd.DataFrame(data["data"], columns=data["columns"])
                        st.table(df)
                        st.session_state.messages.append({"role": "assistant", "type": "table", "content": df})
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

elif not api_key:
    st.warning("Please enter your Google Gemini API Key in the sidebar to proceed.")
elif not uploaded_file:
    st.info("Please upload a CSV file in the sidebar to start chatting.")
