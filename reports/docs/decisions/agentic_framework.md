# Agentic Framework Decision

## Technology Decision

> [!IMPORTANT]
> **Framework Decision Required:** Two viable approaches are presented below. Please confirm your preferred stack before implementation begins.

**Option A: `pydantic-ai` (Recommended for this project)**

| Criteria | Verdict |
|:---|:---|
| Structured output | Native via Pydantic models — zero boilerplate |
| Type safety | First-class, fully compatible with your `pyright` strict mode |
| Tool definitions | Type-annotated functions — Agent reads docstrings per Rule 1.7 |
| State persistence | LangGraph (can be imported separately for checkpointing) |
| Learning curve | Very low — reads like normal Python |
| Portfolio signal | Modern, cutting-edge (2024 framework) |

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class AnalystReport(BaseModel):
    sentiment_summary: str
    trend_analysis: str
    strategic_recommendation: str
    confidence_score: float

agent = Agent("google-gla:gemini-2.0-flash", output_type=AnalystReport)
```

**Option B: `LangChain` + `LangGraph`**

| Criteria | Verdict |
|:---|:---|
| Structured output | Via `PydanticOutputParser` or `.with_structured_output()` |
| Type safety | Weaker — heavy use of `Any` types internally |
| State persistence | Best-in-class (`MemorySaver`, `SqliteSaver`) |
| HITL support | Native `interrupt()` support in LangGraph |
| Learning curve | Higher — verbose, abstraction-heavy |
| Portfolio signal | Industry-standard, widely recognized |

> **Recommendation:** Use **`pydantic-ai`** for the Agent core + tool definitions (clean Python, strict types), and optionally wrap the multi-step orchestration in a lightweight `LangGraph` `StateGraph` if you want state persistence and HITL checkpointing as a second phase. This gives you the best of both worlds.

## LLM Provider Decision

Use **Google Gemini Flash** (`gemini-2.0-flash-lite`) for:
- Cost: ~$0.01 per complex analysis (routing + synthesis)
- Speed: Sub-2s response for structured JSON output
- Integration: Native `pydantic-ai` support via `google-gla` provider
- Alignment: Matches your existing Google toolchain

> [!IMPORTANT]
> **Framework Decision:** Confirm whether you prefer **`pydantic-ai`** (recommended — clean Python, strict types, modern) or **`LangChain/LangGraph`** (industry-standard, better state persistence, higher verbosity) for the Agent core.

> [!IMPORTANT]
> **LLM Provider:** Confirm use of **Google Gemini Flash** (`gemini-2.0-flash-lite`) as the LLM backend. If you have a preference for a different provider (OpenAI, Anthropic), please specify.

## 🛡️ Resilience: Agentic Healing (Error Recovery)

To handle unreliable LLM provider quotas (e.g., Gemini's `429 RESOURCE_EXHAUSTED`), the Analyst Agent implements a **Fallback Strategy**:

1. **Detection**: Catches exceptions containing "429", "RESOURCE_EXHAUSTED", or "quota".
2. **Configuration**: Uses `fallback_enabled` and `fallback_model_name` from `params.yaml`.
3. **Recovery**: Automatically re-instantiates the Agent with a secondary model (**Groq**) and retries the request seamlessly.
4. **Self-Correction**: Uses `retries=3` in the Agent constructor to allow LLMs to fix structured output formatting errors during high-load scenarios.

### Configuration Example (`params.yaml`)
```yaml
agent:
  model_name: "google-gla:gemini-2.0-flash-lite"
  max_comments: 100
  fallback_enabled: true
  fallback_model_name: "groq:llama-3.1-8b-instant"
```

### The Issue
The `Content Intelligence Analyst` was hitting its first provider's quota (Gemini) and correctly switching to the fallback model (Groq). However, the fallback model was still attempting to process **100 comments** (`413 Request too large` error). 

In the Groq free tier, the **Tokens Per Minute (TPM)** limit is typically **12,000**. Fetching 100 comments, combined with the system prompt and the agent's internal reasoning turns, exceeds this 12,000-token limit (the request was measured at **14,991** tokens). Although the fallback logic *instructed* the agent to truncate in the text message, the underlying tool was still hardcoded to fetch the original `max_comments` from the configuration.

### The Fix
I have modified `src/agents/content_analyst.py` to:
1.  **Enforce Physical Truncation**: When switching to fallback, the agent now creates a "reduced configuration" where `max_comments` is explicitly set to **40**.
2.  **Synchronize Tool Execution**: By passing this reduced config to the fallback agent's dependencies, the `fetch_youtube_comments_tool` now actually retrieves only 40 comments, ensuring the subsequent conversation turns stay well within the Groq 12,000-token TPM limit.

### Strategic Value
This multi-provider architecture ensures high availability for the Chrome Extension and API clients. By decoupling probabilistic reasoning (LLMs) from deterministic execution (Tools), we maintain system integrity even across provider failovers.