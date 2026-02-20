# Game Analytics Agent - API

This component exposes the AI agent via a FastAPI endpoint.

## Endpoint

`POST /game/analytics`

### Input
```json
{
  "query": "your question here"
}
```

### Output
```json
{
  "response": "human-readable answer",
  "metadata": {
    "details": "relevant game or statistical data"
  }
}
```

## Features
- **LangChain Integration**: Uses the Pandas DataFrame agent to directly query the cleaned dataset.
- **Groq Support**: Powered by Llama 3 for fast and accurate analytical reasoning.
- **Metadata Extraction**: Returns structured data alongside the textual answer.
