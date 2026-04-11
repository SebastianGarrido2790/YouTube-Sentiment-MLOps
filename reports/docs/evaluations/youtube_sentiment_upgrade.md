# Hybrid Agentic MLOps System

## What Was Built

The YouTube Sentiment Analysis project has been upgraded from a **Hardened MLOps Pipeline (v1.4)** to a **Hybrid Agentic MLOps System (v2.0)**, integrating a `pydantic-ai` "Content Intelligence Analyst" Agent on top of the existing deterministic FTI infrastructure.

---

## Architecture: Brain vs. Brawn

```
                 ┌─────────────────────────────────────────────────────┐
                 │         Chrome Extension (popup.js)                 │
                 │     POST /v1/agent/analyze → AnalystReport JSON     │
                 └─────────────────────┬───────────────────────────────┘
                                       │
                 ┌─────────────────────▼───────────────────────────────┐
                 │   FastAPI Inference Service (src/api/main.py)       │
                 │   └── Agent Router: POST /v1/agent/analyze          │
                 └─────────────────────┬───────────────────────────────┘
                                       │
                 ┌─────────────────────▼───────────────────────────────┐
                 │     Content Intelligence Analyst Agent (Brain)      │
                 │     pydantic-ai + Gemini Flash (gemini-2.0-flash-lite) │
                 │     • Orchestrates tool call sequence                │
                 │     • Synthesizes qualitative narrative ONLY        │
                 └──────┬────────────┬────────────────┬────────────────┘
                        │            │                │
             ┌──────────▼──┐ ┌───────▼──────┐ ┌──────▼──────────────┐
             │ Youtube API │ │ Data Quality │ │  Inference API       │
             │   Tool      │ │   Tool       │ │  Tool                │
             │ (fetch)     │ │ (validate)   │ │  (ML classify)       │
             │ DETERMINISTIC│ │DETERMINISTIC │ │  DETERMINISTIC       │
             └─────────────┘ └──────────────┘ └──────────────────────┘
```

**Core Rule (1.2):** The LLM Agent **never** runs math or classification. It only orchestrates deterministic tools and synthesizes structured narrative.

---

## Files Created

### Agent Layer
| File | Purpose |
|------|---------|
| `src/agents/__init__.py` | Package marker |
| `src/agents/content_analyst.py` | Core pydantic-ai Agent — registers tools, enforces `AnalystReport` return type |
| `src/agents/prompts/system_prompt.py` | Versioned system prompt (Rule 1.5: No "Naked" Prompts) |
| `src/entity/agent_schemas.py` | Pydantic I/O contracts for all agent inputs/outputs |

### Tool Layer (Deterministic Hands)
| File | Purpose |
|------|---------|
| `src/tools/youtube_tool.py` | Fetches YouTube comments via Data API v3, raises `YouTubeToolError` |
| `src/tools/data_quality_tool.py` | Statistical data quality gate — null ratio, batch size, short-comment ratio |
| `src/tools/sentiment_tool.py` | HTTP wrapper around `/v1/predict` — returns typed `SentimentBreakdown` |

### API Layer
| File | Purpose |
|------|---------|
| `src/api/agent_api.py` | FastAPI router — `POST /v1/agent/analyze` |
| `src/api/main.py` (modified) | Mounted `agent_router` into `/v1` namespace |

### Chrome Extension
| File | Purpose |
|------|---------|
| `popup.html` (modified) | Added `🧠 Get AI Analysis` button + AI Report card + AI loading state |
| `popup.js` (modified) | Added `renderAnalystReport()` + AI button handler calling `/v1/agent/analyze` |
| `popup.css` (modified) | AI report card styles (glassmorphism, sentiment pills, recommendation block) |

### Configuration
| File | Purpose |
|------|---------|
| `pyproject.toml` (modified) | Added `pydantic-ai`, `google-genai`, `pytest-asyncio` / `asyncio_mode=auto` |
| `config/config.yaml` (modified) | Added `agent:` config block |
| `src/entity/config_entity.py` (modified) | Added `AgentConfig` Pydantic schema |
| `src/config/configuration.py` (modified) | Added `get_agent_config()` accessor |
| `.env.example` (modified) | Added `GEMINI_API_KEY` |

### Tests
| File | Result |
|------|--------|
| `tests/test_agent_tools.py` | ✅ 16/16 passed (YouTube tool, Data Quality, Sentiment Tool) |
| `tests/test_agent_api.py` | ✅ 8/8 passed (schema validation, 503/502/200 codes) |

---

## Test Results

```
tests/test_agent_tools.py  → 16 passed in 0.32s
tests/test_agent_api.py    → 8  passed in 7.14s
Total: 24 passed            
```

All tests are async-safe via `pytest-asyncio` with `asyncio_mode = "auto"`.

---

## How to Run

### Prerequisites
```bash
# Add to .env
GEMINI_API_KEY=your-key-here
YOUTUBE_API_KEY=your-key-here
```

### Start the Inference API (serves both ML + Agent endpoints)
```bash
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Test the Agent endpoint
```bash
curl -X POST http://127.0.0.1:8000/v1/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ", "max_comments": 50}'
```

Response:
```json
{
  "video_id":"dQw4w9WgXcQ",
  "video_title":"Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)",
  "sentiment_breakdown":{
    "positive_pct":0.7,
    "neutral_pct":0.2,
    "negative_pct":0.1,
    "dominant_sentiment":"positive",
    "total_analyzed":50,
    "raw_predictions":[
      "positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive"
      ]
  },
  "data_quality_passed":true,
  "model_version":"v1.0",
  "executive_summary":"The YouTube video 'Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)' has a significant number of comments, with many users expressing their surprise and amement at being 'Rickrolled'. The comments are generally lighthearted and playful, with some users discussing the song's nostalgic value and others sharing their own experiences of being tricked into watching the video. Overall, the comments suggest that the video remains a beloved and iconic piece of internet culture.",
  "key_insights":[
    "Many users have been 'Rickrolled' and are amused by the experience","The song has nostalgic value for some users","The video remains a iconic piece of internet culture"
  ],
  "strategic_recommendation":"Consider creating content that leverages the nostalgic value of the song and the 'Rickroll' phenomenon, such as a retro-themed music video or a challenge that encourages users to share their own 'Rickroll' experiences.",
  "confidence_score":0.8
}
```

### Run all tests
```bash
uv run pytest tests/test_agent_tools.py tests/test_agent_api.py -v --no-cov
```

### Chrome Extension
Load `chrome-extension/` as an unpacked extension. Open any YouTube video page and click **"🧠 Get AI Analysis"** — the extension calls the Agent endpoint and renders the AnalystReport card in-popup.

I have updated the button in `chrome-extension/popup.html` to have a more descriptive and professional name.

### Change Summary
- **Old Name**: `Analyze Comments`
- **New Name**: `📊 Run Sentiment Analysis`

### Rationale
- **Clarity**: The previous name was generic and didn't distinguish between the deterministic metrics pipeline and the agentic AI pipeline.
- **Consistency**: The new name now pairs perfectly with the purple **"🧠 Get AI Analysis"** button, establishing a clear visual and semantic hierarchy:
    - **Blue Button (📊)**: Triggers the data-driven sentiment dashboard (charts, metrics, wordclouds).
    - **Purple Button (🧠)**: Triggers the agentic AI report (strategic insights, executive summary).
- **Aesthetics**: Added a chart emoji to maintain the modern, "agentic" look-and-feel of the extension UI.

---

## Key Design Decisions

1. **`pydantic-ai` over LangChain/LangGraph** — native Pydantic v2 integration, pyright-compatible, minimal abstraction overhead. LangGraph would've been overengineering for a single-agent linear workflow.

2. **Agent result type = `AnalystReport`** — pydantic-ai enforces this at the framework level. No free-form text can escape the endpoint (Rule 1.4).

3. **Tools registered as closures** — the `build_agent()` factory pattern makes `AgentDeps` config clean injected without global state.

4. **HTTP to Inference API** — the sentiment tool calls `http://127.0.0.1:8000/v1/predict` rather than importing the ML model directly. This preserves microservice decoupling — the Agent would work even if deployed on a different host (Rule 1.3).

5. **Data Quality Gate before ML inference** — a low-quality batch terminates the agentic workflow early with a clear failure reason included in the `AnalystReport.executive_summary`.

---

## Agentic Workflow Sequence

```
User clicks "🧠 Get AI Analysis"
  │
  └─► POST /v1/agent/analyze {video_url, max_comments}
        │
        └─► Agent runs:
              1. fetch_youtube_comments_tool(video_url)  → CommentBatch
              2. check_data_quality_tool(comments)       → DataQualityReport
              3. [if passed] analyze_sentiment_tool()    → SentimentBreakdown
              4. LLM synthesizes "executive_summary", "key_insights",
                 "strategic_recommendation" from structured tool outputs
              5. Returns AnalystReport (validated Pydantic model)
        │
        └─► Chrome Extension renders AnalystReport card
```
