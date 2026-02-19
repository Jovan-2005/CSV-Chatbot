# 🤖 Intelligent CSV Chat & Analysis (Powered by Groq)

This application allows you to chat with any CSV file using natural language. It uses **Groq** (Llama 3.3) for incredibly fast reasoning and **Streamlit** for a modern chat interface.

## 🚀 Features
- **Chat with Data**: Ask questions in plain English (e.g., "What is the average sales?", "How many user churned?").
- **Instant Charts**: Automatically generates bar charts, line graphs, and pie charts using Matplotlib.
- **Smart Tables**: Displays filtered data tables when requested.
- **Universal Support**: Works with *any* CSV file (automatically detects schema).
- **Free & Fast**: Uses the Groq Free Beta API.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM**: Meta Llama 3.3 (via Groq API)
- **Orchestrator**: LangChain
- **Data Analysis**: Pandas (Python Agent)

## 📦 Installation
1. **Clone/Download** this repository.
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Setup API Key**:
   - Rename `.env.example` to `.env` (or create a `.env` file).
   - Add your [Groq API Key](https://console.groq.com/keys):
     ```text
     GROQ_API_KEY=your_key_here
     ```

## ▶️ Usage
1. Run the application:
   ```bash
   streamlit run run.py
   ```
2. **Upload** your CSV file in the sidebar.
3. **Start Chatting**!

## 📂 Project Structure
- `app.py`: Main application code (Streamlit UI).
- `agent.py`: Logic for the LangChain Pandas Agent.
- `.env`: Securely stores your API key.
- `requirements.txt`: Project dependencies.
- `data/`: Folder where uploaded CSVs are stored.
- `chart_image/`: Folder for generated charts.
- `venv/`: Virtual environment folder (Python dependencies).
