# 🤖 Intelligent CSV Chat & Analysis

This application allows you to chat with any CSV file using natural language. It uses **Groq** for high-speed reasoning, **LangChain** for agent orchestration, and **Streamlit** for a modern, responsive chat interface.

## 🚀 Key Features

-   **Chat with Data**: Ask questions in plain English (e.g., "What are the top 5 records?", "Summarize the average rating by category").
-   **Configurable Models**: Switch between `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` via environment variables.
-   **Robust Visualization**: Automatically generates and displays charts and tables.
-   **Infinite Loop Protection**: Built-in iteration limits and repetition guards to ensure the agent always provides a response.
-   **Smart Data Parsing**: Advanced logic to extract and render tables even if the LLM output is not perfectly formatted.
-   **Performance Caching**: Intelligent session management so you only initialize the AI agent once per file.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **LLM**: Meta Llama 3 via Groq API
-   **Orchestrator**: LangChain (Pandas DataFrame Agent)
-   **Data Analysis**: Pandas

## 📦 Installation & Setup

1.  **Clone the repository**.
2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/Mac
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```text
    GROQ_API_KEY=your_groq_api_key
    MODEL_NAME=llama-3.1-8b-instant
    ```

## ▶️ Usage

1.  Start the application:
    ```bash
    streamlit run app.py
    ```
2.  **Upload** your CSV file in the sidebar.
3.  **Start Chatting**!

## 📂 Project Structure

-   `app.py`: Main Streamlit application and UI logic.
-   `agent.py`: LangChain agent configuration and prompt engineering.
-   `.env`: Secure configuration for API keys and model selection.
-   `data/`: Secure storage for uploaded files.
-   `chart_image/`: Temporary storage for generated visualizations.
