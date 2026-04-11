// ==========================================================
// YouTube Sentiment Chrome Extension - Insights
// ==========================================================

import { fetchYouTubeComments } from "./youtube_api.js";

// ========================
// CONFIGURATION
// ========================
let currentApiKey = "";
const INSIGHTS_URL = "http://127.0.0.1:8001";                   // Insights API base URL (port 8001)

// ========================
// DOM ELEMENTS
// ========================
const analyzeBtn = document.getElementById("analyzeBtn");
const numCommentsInput = document.getElementById("numComments");
const loadingEl = document.getElementById("loading");
const resultsEl = document.getElementById("results");
const totalCountEl = document.getElementById("totalCount");
const breakdownText = document.getElementById("breakdownText");
const pieChartImg = document.getElementById("pieChartImg");     // <img> for pie chart
const legendEl = document.getElementById("legend");
const errorEl = document.getElementById("error");
const backendUrlEl = document.getElementById("backendUrl");
const loadingTextEl = document.getElementById("loadingText");
const videoIdEl = document.getElementById("videoId");           // Video ID display
const videoStatusEl = document.getElementById("video-status"); // Video status label

// Settings elements
const apiKeyInput = document.getElementById("apiKey");
const saveKeyBtn = document.getElementById("saveKeyBtn");
const keyStatusEl = document.getElementById("keyStatus");

// Summary metrics
const totalCommentsSummaryEl = document.getElementById("totalCommentsSummary");
const uniqueCommentersEl = document.getElementById("uniqueCommenters");
const avgLenEl = document.getElementById("avgLen");
const avgSentimentEl = document.getElementById("avgSentiment");

// Advanced insights elements (assume added to HTML)
const trendGraphImg = document.getElementById("trendGraphImg"); // <img> for trend graph
const wordcloudImg = document.getElementById("wordcloudImg");   // <img> for wordcloud
const topCommentsEl = document.getElementById("topComments");   // <ul> or <div> for top 25

backendUrlEl.textContent = INSIGHTS_URL;

// ========================
// INITIALIZATION
// ========================
async function init() {
  const result = await chrome.storage.local.get(["yt_api_key"]);
  if (result.yt_api_key) {
    currentApiKey = result.yt_api_key;
    if (apiKeyInput) apiKeyInput.value = currentApiKey;
    if (keyStatusEl) {
      keyStatusEl.textContent = "Key loaded from storage";
      keyStatusEl.classList.remove("hidden");
    }
  }
}

init();

saveKeyBtn.addEventListener("click", async () => {
  const key = apiKeyInput.value.trim();
  if (!key) {
    showError("Please enter a valid API Key.");
    return;
  }
  await chrome.storage.local.set({ yt_api_key: key });
  currentApiKey = key;
  keyStatusEl.textContent = "Key saved!";
  keyStatusEl.classList.remove("hidden");
  setTimeout(() => keyStatusEl.classList.add("hidden"), 3000);
});

// ========================
// UI HELPERS
// ========================
function showLoading(text = "Analyzing comments...") {
  loadingTextEl.innerText = text;
  loadingEl.classList.remove("hidden");
  resultsEl.classList.add("hidden");
  errorEl.classList.add("hidden");
  analyzeBtn.disabled = true;
}

function hideLoading() {
  loadingEl.classList.add("hidden");
  analyzeBtn.disabled = false;
}

function showError(msg) {
  errorEl.innerText = msg;
  errorEl.classList.remove("hidden");
  resultsEl.classList.add("hidden");
  loadingEl.classList.add("hidden");
  analyzeBtn.disabled = false;
}

function displayImage(imgEl, blob) {
  const url = URL.createObjectURL(blob);
  imgEl.src = url;
  imgEl.style.display = "block";
}

// ========================
// YOUTUBE COMMENT FETCH
// ========================
function getVideoIdFromUrl(url) {
  try {
    const params = new URL(url).searchParams;
    return params.get("v");
  } catch {
    return null;
  }
}

async function getCommentsFromAPI(maxResults = 100) {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (!tabs || !tabs.length) return reject("No active YouTube tab found.");
      const tab = tabs[0];
      const videoId = getVideoIdFromUrl(tab.url);
      if (!videoId) return reject("No video detected. Please open a YouTube video.");

      // Display Video ID
      if (videoIdEl) {
        videoIdEl.textContent = `(ID: ${videoId})`;
        videoIdEl.style.display = "inline";
      }
      if (videoStatusEl) videoStatusEl.textContent = "Video detected";

      try {
        if (!currentApiKey) {
          return reject("API Key missing. Please enter and save your YouTube API Key first.");
        }
        const comments = await fetchYouTubeComments(videoId, currentApiKey, maxResults);
        resolve(comments);  // Now [{text, timestamp}]
      } catch (err) {
        reject(err);
      }
    });
  });
}

// ========================
// INSIGHTS API CALLS
// ========================
async function callPredictWithTimestamps(commentsData) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000); // 30s timeout

  try {
    const resp = await fetch(`${INSIGHTS_URL}/v1/predict_with_timestamps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: commentsData }),
      signal: controller.signal,
    });

    clearTimeout(timeout);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Insights API error (${resp.status}): ${text}`);
    }

    return await resp.json();
  } catch (err) {
    if (err.name === "AbortError") throw new Error("API timed out (30s).");
    throw err;
  }
}

async function generateChart(sentimentCounts) {
  const resp = await fetch(`${INSIGHTS_URL}/v1/generate_chart`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentiment_counts: sentimentCounts }),
  });
  if (!resp.ok) throw new Error("Failed to generate pie chart.");
  return await resp.blob();
}

async function generateWordcloud(comments) {
  const resp = await fetch(`${INSIGHTS_URL}/v1/generate_wordcloud`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comments }),
  });
  if (!resp.ok) throw new Error("Failed to generate wordcloud.");
  return await resp.blob();
}

async function generateTrendGraph(sentimentData) {
  const resp = await fetch(`${INSIGHTS_URL}/v1/generate_trend_graph`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentiment_data: sentimentData }),
  });
  if (!resp.ok) throw new Error("Failed to generate trend graph.");
  return await resp.blob();
}

// ========================
// ANALYTICS HELPERS
// ========================
function normalizeSentiments(sentiments) {
  // Map numeric (-1,0,1) to strings
  return sentiments.map(s => {
    if (s === -1) return "Negative";
    if (s === 0) return "Neutral";
    if (s === 1) return "Positive";
    return "Unknown";
  });
}

function computeSentimentCounts(sentiments) {
  const counts = { Positive: 0, Neutral: 0, Negative: 0 };
  sentiments.forEach(s => {
    if (s === 1) counts.Positive++;
    else if (s === 0) counts.Neutral++;
    else if (s === -1) counts.Negative++;
  });
  return counts;
}

function updateLegendAndBreakdown(counts) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const labels = ["Positive", "Neutral", "Negative"];
  legendEl.innerHTML = "";

  labels.forEach(label => {
    const count = counts[label] || 0;
    const pct = total ? ((count / total) * 100).toFixed(1) : "0.0";
    const color = label === "Positive" ? "#10b981" : label === "Neutral" ? "#9ca3af" : "#ef4444";

    const item = document.createElement("div");
    item.classList.add("item");
    item.innerHTML = `
      <span class="swatch" style="background:${color}"></span>
      ${label}: <strong>${count}</strong> (${pct}%)
    `;
    legendEl.appendChild(item);
  });

  breakdownText.innerHTML = `
    Positive <strong>${counts["Positive"] || 0}</strong> —
    Neutral <strong>${counts["Neutral"] || 0}</strong> —
    Negative <strong>${counts["Negative"] || 0}</strong>
  `;
}

function displayTopComments(results, limit = 25) {
  if (!topCommentsEl) return;
  topCommentsEl.innerHTML = "";
  const sorted = results.slice(0, limit).map(r => ({
    ...r,
    label: normalizeSentiments([r.sentiment])[0]
  }));
  sorted.forEach((item, i) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <strong>${i + 1}.</strong> ${item.comment.substring(0, 100)}... 
      <span style="color: ${item.sentiment === 1 ? '#10b981' : item.sentiment === 0 ? '#9ca3af' : '#ef4444'}">
        (Sentiment: ${item.sentiment})
      </span>
    `;
    topCommentsEl.appendChild(li);
  });
}

// ========================
// MAIN WORKFLOW
// ========================
analyzeBtn.addEventListener("click", async () => {
  showLoading("Fetching comments from YouTube...");

  try {
    const limit = Math.max(1, Number(numCommentsInput.value) || 100);

    // Step 1: Fetch comments with timestamps
    const commentsData = await getCommentsFromAPI(limit);
    if (!commentsData || commentsData.length === 0)
      throw new Error("No comments retrieved.");

    showLoading(`Analyzing ${commentsData.length} comments...`);

    // Step 2: Predict with timestamps
    const resp = await callPredictWithTimestamps(commentsData);
    const results = resp.results || [];
    if (!results.length) throw new Error("No predictions returned.");

    // Step 3: Extract sentiments (numeric -1/0/1)
    const sentiments = results.map(r => r.sentiment);
    const stringSentiments = normalizeSentiments(sentiments);

    // Step 4: Compute metrics
    const counts = computeSentimentCounts(sentiments);
    const totalComments = commentsData.length;
    const uniqueCommenters = new Set(commentsData.map(c => c.author || "")).size; // Assume author if available; adjust if needed
    const totalWords = commentsData.reduce(
      (sum, c) => sum + c.text.split(/\s+/).filter(w => w.length > 0).length,
      0
    );
    const avgCommentLength = (totalWords / totalComments).toFixed(2);

    const sentimentMap = { Positive: 1, Neutral: 0, Negative: -1 };
    const avgSentimentRaw = stringSentiments
      .map(p => sentimentMap[p] ?? 0)
      .reduce((a, b) => a + b, 0) / totalComments;
    const avgSentimentScore = (((avgSentimentRaw + 1) / 2) * 10).toFixed(2);

    // Step 5: Update basic UI
    totalCountEl.innerText = totalComments;
    totalCommentsSummaryEl.innerText = totalComments;
    uniqueCommentersEl.innerText = uniqueCommenters;
    avgLenEl.innerText = avgCommentLength;
    avgSentimentEl.innerText = avgSentimentScore;

    updateLegendAndBreakdown(counts);

    // Step 6: Generate and display visuals
    showLoading("Generating visuals...");

    // Pie Chart
    const sentimentCounts = {
      "-1": counts.Negative,
      "0": counts.Neutral,
      "1": counts.Positive
    };
    const chartBlob = await generateChart(sentimentCounts);
    displayImage(pieChartImg, chartBlob);

    // Wordcloud
    const texts = commentsData.map(c => c.text);
    const wordcloudBlob = await generateWordcloud(texts);
    displayImage(wordcloudImg, wordcloudBlob);

    // Trend Graph
    const sentimentData = results.map(r => ({ sentiment: r.sentiment, timestamp: r.timestamp }));
    const trendBlob = await generateTrendGraph(sentimentData);
    displayImage(trendGraphImg, trendBlob);

    // Top Comments
    displayTopComments(results);

    resultsEl.classList.remove("hidden");
    loadingTextEl.innerText = "Analysis complete!";
  } catch (err) {
    console.error("[YouTube Sentiment] Error:", err);
    showError(err.message || String(err));
  } finally {
    hideLoading();
  }
});

// ========================
// AI ANALYST INTEGRATION
// ========================
const INFERENCE_URL = "http://127.0.0.1:8000";
const aiAnalyzeBtn = document.getElementById("aiAnalyzeBtn");
const aiReportEl = document.getElementById("ai-report");
const aiLoadingEl = document.getElementById("ai-loading");

/**
 * Shows the AI loading state, hiding other sections.
 */
function showAiLoading() {
  aiLoadingEl.classList.remove("hidden");
  aiReportEl.classList.add("hidden");
  errorEl.classList.add("hidden");
  if (aiAnalyzeBtn) aiAnalyzeBtn.disabled = true;
}

/**
 * Hides the AI loading state.
 */
function hideAiLoading() {
  aiLoadingEl.classList.add("hidden");
  if (aiAnalyzeBtn) aiAnalyzeBtn.disabled = false;
}

/**
 * Renders an AnalystReport JSON object into the AI report section.
 * @param {Object} report - AnalystReport from POST /v1/agent/analyze
 */
function renderAnalystReport(report) {
  const titleEl = document.getElementById("ai-report-title");
  const confidenceBadge = document.getElementById("ai-confidence-badge");
  const qualityGateEl = document.getElementById("ai-quality-gate");
  const pillsEl = document.getElementById("ai-sentiment-pills");
  const summaryEl = document.getElementById("ai-executive-summary");
  const insightsEl = document.getElementById("ai-key-insights");
  const recommendationEl = document.getElementById("ai-recommendation");
  const modelVersionEl = document.getElementById("ai-model-version");

  // Title
  if (titleEl) titleEl.textContent = report.video_title || "Content Intelligence Report";

  // Confidence badge
  if (confidenceBadge) {
    const pct = Math.round((report.confidence_score || 0) * 100);
    const color = pct >= 70 ? "#10b981" : pct >= 40 ? "#f59e0b" : "#ef4444";
    confidenceBadge.textContent = `${pct}% confidence`;
    confidenceBadge.style.background = `${color}22`;
    confidenceBadge.style.color = color;
    confidenceBadge.style.border = `1px solid ${color}44`;
  }

  // Data quality gate
  if (qualityGateEl) {
    const passed = report.data_quality_passed;
    qualityGateEl.textContent = passed ? "✅ Data quality gate passed" : "⚠️ Data quality issues detected";
    qualityGateEl.style.color = passed ? "#10b981" : "#f59e0b";
  }

  // Sentiment pills
  if (pillsEl && report.sentiment_breakdown) {
    const bd = report.sentiment_breakdown;
    const pills = [
      { label: "Positive", pct: bd.positive_pct, color: "#10b981" },
      { label: "Neutral", pct: bd.neutral_pct, color: "#9ca3af" },
      { label: "Negative", pct: bd.negative_pct, color: "#ef4444" },
    ];
    pillsEl.innerHTML = pills
      .map(p => `
        <span class="sentiment-pill" style="background:${p.color}22; color:${p.color}; border:1px solid ${p.color}44;">
          ${p.label} ${Math.round(p.pct * 100)}%
        </span>
      `)
      .join("");
  }

  // Executive summary
  if (summaryEl) summaryEl.textContent = report.executive_summary || "";

  // Key insights
  if (insightsEl) {
    insightsEl.innerHTML = (report.key_insights || [])
      .map(insight => `<li>${insight}</li>`)
      .join("");
  }

  // Strategic recommendation
  if (recommendationEl) recommendationEl.textContent = report.strategic_recommendation || "";

  // Model version
  if (modelVersionEl) modelVersionEl.textContent = report.model_version || "unknown";

  // Show the section
  aiReportEl.classList.remove("hidden");
}

/**
 * Main handler for the "Get AI Analysis" button.
 * Calls POST /v1/agent/analyze on the Inference API with the current video URL.
 */
aiAnalyzeBtn && aiAnalyzeBtn.addEventListener("click", async () => {
  showAiLoading();

  try {
    // Get the current tab URL
    const tabs = await new Promise(resolve =>
      chrome.tabs.query({ active: true, currentWindow: true }, resolve)
    );
    const tab = tabs && tabs[0];
    if (!tab || !tab.url) throw new Error("No active tab detected.");

    const videoUrl = tab.url;
    if (!videoUrl.includes("youtube.com/watch")) {
      throw new Error("Please open a YouTube video page before running AI Analysis.");
    }

    const maxComments = Math.min(500, Math.max(10, Number(numCommentsInput.value) || 100));

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90000); // 90s for agentic workflow

    const resp = await fetch(`${INFERENCE_URL}/v1/agent/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_url: videoUrl, max_comments: maxComments }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`Agent API error (${resp.status}): ${errText}`);
    }

    const report = await resp.json();
    renderAnalystReport(report);

  } catch (err) {
    if (err.name === "AbortError") {
      showError("AI Analysis timed out (90s). The agent may still be processing — try again.");
    } else {
      console.error("[AI Analyst] Error:", err);
      showError(err.message || String(err));
    }
  } finally {
    hideAiLoading();
  }
});