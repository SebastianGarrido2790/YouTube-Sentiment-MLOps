# Hybrid Agentic MLOps — Strategic Assessment & Implementation Plan

**Project:** YouTube Sentiment Analysis  
**Current State:** Hardened MLOps Ecosystem (9.2 / 10)  
**Proposed Transition:** Standard MLOps Pipeline → Hybrid Agentic MLOps System

---

## Executive Summary

The codebase is already at **elite production quality** (9.2/10). The infrastructure — deterministic DVC pipeline, dual FastAPI services, Great Expectations data contracts, MLflow model registry — is precisely the "hands" that an Agentic Brain needs to orchestrate. The FTI architecture is not just compatible with an agentic overlay; it was *designed for it*. The question is not *whether* to add an agentic layer — it is **which agentic pattern to implement and how to wire it to the existing deterministic infrastructure**.

This is a high-leverage move. Adding an agent on top of a fragile pipeline creates chaos. Adding an agent on top of **your** pipeline creates a showcase system.

---

## 1. Why Now & Why This Project

### 1.1 The Portfolio Gap This Closes

Most ML portfolio projects fall into two anti-patterns:

| Anti-Pattern | What It Signals to Employers |
|:---|:---|
| Jupyter notebook → deployed model | "Can follow a tutorial" |
| MLOps pipeline with DVC + MLflow | "Strong engineer, but executes manually" |
| **Hybrid Agentic MLOps System** | **"Thinks in systems, builds autonomously operating pipelines"** |

Your current 9.2/10 project demonstrates the second tier. The agentic layer is the leap to the third tier — and it reflects the *actual direction the industry is moving*.

### 1.2 Strategic Fit With Your Codebase

Your existing architecture already provides **all the deterministic tools the Agent needs:**

```
Existing Asset                         → Becomes Agent Tool
─────────────────────────               ─────────────────────────────────────────
DVC pipeline (12 stages)             → Pipeline Orchestration Tool
FastAPI /v1/predict (port 8000)      → Sentiment Inference Tool
FastAPI /v1/insights (port 8001)     → Trend Analysis Tool
Great Expectations validation        → Data Quality Audit Tool
MLflow Registry (champion selection) → Model Registry Tool
YouTube Data API v3 (ingestion)      → Live Data Fetching Tool
Chrome Extension (UI layer)          → Human-in-the-Loop Interface
```

You are not adding complexity. You are adding **intelligence** that *directs* the pipeline that already exists.

---

## 2. Selected Agentic Pattern: The Content Intelligence Analyst

### 2.1 The Business Narrative

> *"A content creator or brand manager enters a YouTube video URL. Instead of receiving a raw pie chart of sentiment percentages, they receive a fully curated analyst report: 'Your last 3 videos show a 22% decline in positive sentiment, concentrated in comments about video pacing. Your audience responds best to hands-on tutorials. Competitors in your niche are gaining ground on "beginner-friendly" positioning. Recommended action: A/B test a 10-minute format for your next 3 uploads.'"*

This is the **qualitative synthesis layer** that: (a) no pure ML pipeline can produce, (b) directly maps to ROI for any content creator or brand.

### 2.2 Recommended Pattern: Sequential Agent with Tool-Calling

Based on your workflow and the deterministic nature of your pipeline, the **Sequential Agent Pattern** is the ideal fit:

```
    User Request (video URL)
         │
         ▼
    ┌─────────────────────────────────────────┐
    │   Content Intelligence Analyst Agent    │  ← pydantic-ai Agent (The Brain)
    │   (LangGraph StateGraph + checkpointer) │
    └──────────┬────────────────────────────┬─┘
               │                            │
    ┌──────────▼──────┐          ┌──────────▼──────────┐
    │  Data Fetching  │          │  Sentiment Analysis  │
    │  Tool           │          │  Tool                │
    │  (YouTube API)  │          │  (FastAPI /predict)  │
    └──────────┬──────┘          └──────────┬───────────┘
               │                            │
    ┌──────────▼──────┐          ┌──────────▼──────────┐
    │  Trend Analysis │          │  Data Quality Gate   │
    │  Tool           │          │  Tool                │
    │  (FastAPI       │          │  (GX Validation)     │
    │   /insights)    │          │                      │
    └──────────┬──────┘          └──────────┬───────────┘
               │                            │
               └────────────┬───────────────┘
                            │
                   ┌────────▼──────────┐
                   │  Synthesis Tool   │
                   │  (LLM structured  │
                   │   output → JSON)  │
                   └────────┬──────────┘
                            │
                   ┌────────▼──────────┐
                   │  HITL Gate        │  ← Optional: human approval before delivery
                   │  (Interrupt node) │
                   └────────┬──────────┘
                            │
                   ┌────────▼──────────┐
                   │  Analyst Report   │
                   │  (Pydantic Schema)│
                   └───────────────────┘
```

---

## 3. Technical Architecture

> **Confirmed Decisions:**
> - Framework: pydantic-ai
> - LLM Provider: Google Gemini Flash (gemini-2.0-flash-lite)

### 3.1 New Agent Module Structure

The following new files integrate into your existing `src/` hierarchy without breaking anything:

```
src/
├── agents/                          # NEW: Agentic Brain Layer
│   ├── __init__.py
│   ├── content_analyst.py           # Core pydantic-ai agent definition
│   └── prompts/
│       ├── __init__.py
│       └── system_prompt.py         # Versioned, templated system prompts (Rule 1.5)
│
├── tools/                           # RENAMED/EXTENDED: Deterministic Hands
│   ├── __init__.py
│   ├── dvc/                         # DVC Pipeline scripts (Comparison, Tuning)
│   │   ├── feature_comparison.py
│   │   └── ...
│   ├── sentiment_tool.py            # Wraps FastAPI /v1/predict
│   ├── insights_tool.py             # Wraps FastAPI /v1/insights
│   ├── youtube_tool.py              # Wraps YouTube Data API v3
│   └── data_quality_tool.py         # Wraps GX validation check
│
├── entity/
│   └── agent_schemas.py             # NEW: Pydantic I/O contracts for Agent
│
└── api/
    └── agent_api.py                 # NEW: FastAPI endpoint exposing the Agent
```

### 3.2 The Agent's Data Contracts (Rule 1.4)

```python
# src/entity/agent_schemas.py

class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_url: str = Field(..., description="YouTube video URL to analyze")
    max_comments: int = Field(default=100, ge=10, le=500)
    language: str = Field(default="en")

class SentimentBreakdown(BaseModel):
    positive_pct: float = Field(..., ge=0.0, le=1.0)
    neutral_pct: float = Field(..., ge=0.0, le=1.0)
    negative_pct: float = Field(..., ge=0.0, le=1.0)
    dominant_themes: list[str]

class AnalystReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_id: str
    sentiment_breakdown: SentimentBreakdown
    executive_summary: str          # 2-3 sentence business narrative
    key_insights: list[str]         # Bullet-point findings
    strategic_recommendation: str  # Single actionable directive
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    data_quality_passed: bool
    model_version: str
```

### 3.3 Tool Design (Rule 1.3 — Tools as Microservices)

All tools are **deterministic, typed wrappers** that the Agent calls. The LLM never does math.

```python
# src/tools/sentiment_tool.py

async def analyze_sentiment(comments: list[str]) -> SentimentBreakdown:
    """
    Calls the Inference API to classify a batch of YouTube comments.

    Sends raw comment text to the /v1/predict endpoint and returns
    a structured SentimentBreakdown with percentage splits by class.
    The LLM must NEVER attempt to classify sentiment directly.

    Args:
        comments: List of raw YouTube comment strings (max 500).

    Returns:
        SentimentBreakdown with positive/neutral/negative percentages.

    Raises:
        InferenceAPIError: If the prediction service is unreachable.
    """
    ...
```

### 3.4 Agent API Endpoint

A new FastAPI route exposes the Agent as an API, callable from the Chrome Extension:

```
POST /v1/agent/analyze
Body: { "video_url": "https://youtube.com/watch?v=...", "max_comments": 100 }
Response: AnalystReport (structured JSON)
```

This wires directly into the existing Chrome Extension as a new "AI Analysis" tab.

---

## 4. Integration with Chrome Extension

The Chrome Extension becomes the **Human-in-the-Loop interface**:

- Existing: Raw pie charts + sentiment percentages
- New: "Get AI Analysis" button → calls `/v1/agent/analyze` → returns an `AnalystReport` card with the strategic narrative rendered in the popup

No structural change to the extension is required — only an additional API call and a new render function in `popup.js`.

---

## 5. Proposed Changes

### Phase A: Foundation

#### [NEW] `src/agents/__init__.py`
#### [NEW] `src/agents/content_analyst.py`
The core `pydantic-ai` Agent with tool registration and result type set to `AnalystReport`.

#### [NEW] `src/agents/prompts/system_prompt.py`
Versioned, templated system prompt following Rule 1.5. No naked strings.

#### [NEW] `src/entity/agent_schemas.py`
Pydantic I/O data contracts: `AnalysisRequest`, `SentimentBreakdown`, `AnalystReport`.

---

### Phase B: Tools (Deterministic Hands)

#### [MODIFY] `src/tools/__init__.py`
#### [NEW] `src/tools/sentiment_tool.py`
HTTP wrapper around `src/api/main.py` (`POST /v1/predict`). Fully typed, raises `InferenceAPIError` on failure.

#### [NEW] `src/tools/insights_tool.py`
HTTP wrapper around `src/api/insights_api.py` (`POST /v1/insights/...`). Returns chart data as structured Pydantic models.

#### [NEW] `src/tools/youtube_tool.py`
Typed wrapper around the YouTube Data API v3. Accepts a video URL, extracts the video ID, fetches top comments, returns `list[str]`.

#### [NEW] `src/tools/data_quality_tool.py`
Calls GX validation check on the fetched comment batch. Returns a `bool` pass/fail with a metadata summary.

---

### Phase C: Agent API Exposure

#### [NEW] `src/api/agent_api.py`
FastAPI router exposing `POST /v1/agent/analyze`. Registered in `src/api/main.py`.

#### [MODIFY] `src/api/main.py`
Include the new `agent_router` alongside existing routers.

---

### Phase D: Chrome Extension

#### [MODIFY] `chrome-extension/popup.js`
Add a "Get AI Analysis" button and a new render function for the `AnalystReport` card.

#### [MODIFY] `chrome-extension/popup.html`
Add the AI Analysis tab/section to the UI.

---

### Phase E: Configuration & Dependencies

#### [MODIFY] `pyproject.toml`
Add `pydantic-ai`, `google-generativeai` (or `pydantic-ai[google]`), and `httpx` (async HTTP for tool calls).

#### [MODIFY] `.env.example`
Add `GEMINI_API_KEY=your_gemini_api_key_here`.

#### [MODIFY] `config/config.yaml`
Add agent configuration block: `model_name`, `max_tokens`, `tool_timeout_seconds`.

#### [MODIFY] `src/config/schemas.py`
Add `AgentConfig` Pydantic schema.

---

## 6. What This Does NOT Change

> [!IMPORTANT]
> The existing DVC pipeline, MLflow integration, Great Expectations validation, and all deterministic components are **completely untouched**. The Agent is a consumer of these systems, not a replacement for them. Zero regression risk.

---

## 7. MLOps Maturity Impact

| Dimension | Before Agent | After Agent |
|:---|:---:|:---:|
| Codebase Score | 9.2 / 10 | ~9.7 / 10 |
| Positioning | "Senior MLOps Engineer" | "Agentic ML Systems Architect" |
| Business Value Demo | Technical dashboard | Executive analyst report |
| Architecture Tier | FTI Pipeline | Hybrid Agentic MLOps System |
| Portfolio Uniqueness | High | **Exceptional** |

---

## 8. Verification Plan

### Automated Tests
- `tests/test_agent_tools.py` — unit tests for each tool mocking the downstream API calls
- `tests/test_agent_api.py` — FastAPI `TestClient` test for `POST /v1/agent/analyze`
- Updated `validate_system.bat` — Pillar 4 extended to include Agent API health check

### Manual Verification
- End-to-end: Enter a YouTube video URL → receive full `AnalystReport` via Chrome Extension popup
- Confirm structured JSON response matches `AnalystReport` Pydantic schema exactly
- Confirm tools never pass math or classification directly to the LLM

---

> [!NOTE]
> **HITL Scope:** The initial implementation will not include a human-in-the-loop interrupt node (to keep scope controlled). This can be added in a Phase 2 iteration using LangGraph checkpointing if desired.
