# Panda Dive Local Demo

Quick local demo with a polished single-page UI and a tiny Python server.

## Run

1. Ensure your `.env` contains model/search related API keys (based on your selected model).
2. Start server from repo root:

```bash
python local_demo/server.py
```

3. Open `http://127.0.0.1:8787`.

## API

- `GET /api/health`
- `POST /api/research`

Example request:

```json
{
  "topic": "2026 agentic AI enterprise landscape",
  "settings": {
    "search_api": "duckduckgo",
    "max_researcher_iterations": 4,
    "max_concurrent_research_units": 3,
    "allow_clarification": false,
    "query_variants": 3,
    "relevance_threshold": 0.7,
    "rerank_top_k": 10
  }
}
```
