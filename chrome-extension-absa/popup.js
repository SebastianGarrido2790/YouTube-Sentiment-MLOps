// ==========================================================
// YouTube Sentiment Chrome Extension - ABSA Version
// ==========================================================

import { fetchYouTubeComments } from "./youtube_api.js";

// ========================
// CONFIGURATION
// ========================
const API_KEY = "AIzaSyA3cnCtBOXx_6G8zvxm3Y-OFpjRWD7I_VU"; // Replace with your YouTube API key
const SENTIMENT_API_URL = "http://127.0.0.1:8000";
const ASPECTS_TO_ANALYZE = ["video quality", "audio", "content", "presenter"];

// ========================
// DOM ELEMENTS
// ========================
const analyzeBtn = document.getElementById("analyzeBtn");
const numCommentsInput = document.getElementById("numComments");
const loadingEl = document.getElementById("loading");
const resultsEl = document.getElementById("results");
const absaResultsEl = document.getElementById("absa-results");
const errorEl = document.getElementById("error");
const backendUrlEl = document.getElementById("backendUrl");
const loadingTextEl = document.getElementById("loadingText");
const videoIdEl = document.getElementById("videoId");

backendUrlEl.textContent = SENTIMENT_API_URL;

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

async function getCommentsFromAPI(maxResults = 20) {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (!tabs || !tabs.length) return reject("No active YouTube tab found.");
      const tab = tabs[0];
      const videoId = getVideoIdFromUrl(tab.url);
      if (!videoId) return reject("No video detected. Please open a YouTube video.");

      if (videoIdEl) videoIdEl.textContent = videoId;

      try {
        const comments = await fetchYouTubeComments(videoId, API_KEY, maxResults);
        resolve(comments);
      } catch (err) {
        reject(err);
      }
    });
  });
}

// ========================
// SENTIMENT API CALL
// ========================
async function callABSAPredict(commentText, aspects) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000); // 15s timeout

  try {
    const resp = await fetch(`${SENTIMENT_API_URL}/predict_absa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: commentText, aspects: aspects }),
      signal: controller.signal,
    });

    clearTimeout(timeout);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`API error (${resp.status}): ${text}`);
    }

    return await resp.json();
  } catch (err) {
    if (err.name === "AbortError") throw new Error("API timed out (15s).");
    throw err;
  }
}

// ========================
// RESULTS DISPLAY
// ========================
function displayABSAResults(results) {
  absaResultsEl.innerHTML = ""; // Clear previous results

  if (!results || results.length === 0) {
    absaResultsEl.innerHTML = "<p>No aspects were found in the comments.</p>";
    return;
  }

  results.forEach(result => {
    const commentCard = document.createElement("div");
    commentCard.className = "absa-comment";

    const commentTextEl = document.createElement("p");
    commentTextEl.className = "absa-comment-text";
    commentTextEl.textContent = result.comment;

    const aspectsListEl = document.createElement("ul");
    aspectsListEl.className = "absa-aspects-list";

    result.analysis.forEach(aspect => {
      const aspectItemEl = document.createElement("li");
      aspectItemEl.className = "absa-aspect-item";

      const aspectName = document.createElement("span");
      aspectName.className = "absa-aspect-name";
      aspectName.textContent = aspect.aspect;

      const aspectSentiment = document.createElement("span");
      aspectSentiment.className = `absa-aspect-sentiment ${aspect.sentiment.toLowerCase()}`;
      aspectSentiment.textContent = aspect.sentiment;

      aspectItemEl.appendChild(aspectName);
      aspectItemEl.appendChild(aspectSentiment);
      aspectsListEl.appendChild(aspectItemEl);
    });

    commentCard.appendChild(commentTextEl);
    commentCard.appendChild(aspectsListEl);
    absaResultsEl.appendChild(commentCard);
  });
}

// ========================
// MAIN WORKFLOW
// ========================
analyzeBtn.addEventListener("click", async () => {
  showLoading("Fetching comments from YouTube...");

  try {
    const limit = Math.max(1, Number(numCommentsInput.value) || 20);
    const commentsData = await getCommentsFromAPI(limit);
    if (!commentsData || commentsData.length === 0) {
      throw new Error("No comments were retrieved from the video.");
    }

    const allResults = [];
    let processedCount = 0;

    for (const comment of commentsData) {
      processedCount++;
      showLoading(`Analyzing comment ${processedCount} of ${commentsData.length}...`);
      try {
        const analysis = await callABSAPredict(comment.text, ASPECTS_TO_ANALYZE);
        
        // Only include results where at least one aspect was non-neutral
        const significantAnalysis = analysis.filter(a => a.sentiment.toLowerCase() !== 'neutral');

        if (significantAnalysis.length > 0) {
            allResults.push({
              comment: comment.text,
              analysis: significantAnalysis,
            });
        }
      } catch (error) {
        console.warn(`Could not analyze comment: "${comment.text.substring(0, 50)}..."`, error);
        // Continue to the next comment even if one fails
      }
    }

    displayABSAResults(allResults);

    resultsEl.classList.remove("hidden");
  } catch (err) {
    console.error("[YouTube Sentiment] Error:", err);
    showError(err.message || String(err));
  } finally {
    hideLoading();
  }
});
