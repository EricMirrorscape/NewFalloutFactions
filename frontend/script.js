const API_URL = "/chat";

document.addEventListener("DOMContentLoaded", () => {
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const conversationArea = document.getElementById("conversation-area");
  const welcomeMessage = document.getElementById("welcome-message");

  let sessionId = null;
  let sending = false;

  function scrollToBottom() {
    conversationArea.scrollTop = conversationArea.scrollHeight;
  }

  function addUserMessage(text) {
    const userDiv = document.createElement("div");
    userDiv.className = "user-message";
    userDiv.innerHTML = `
      <div class="user-label">YOU</div>
      <div class="user-text">${escapeHtml(text)}</div>
    `;
    conversationArea.appendChild(userDiv);
    scrollToBottom();
  }

  function addBotResponse(shortAnswer, reasoning, sources) {
    const botDiv = document.createElement("div");
    botDiv.className = "bot-response";
    
    let sourcesHtml = "";
    if (sources && sources.length > 0) {
      const sourceTags = sources.map(s => 
        `<span class="source-tag">Page ${s.page}</span>`
      ).join("");
      sourcesHtml = `
        <div class="sources-section">
          <span class="sources-label">Sources:</span>
          ${sourceTags}
        </div>
      `;
    }
    
    botDiv.innerHTML = `
      <div class="bot-label">BOT RESPONSE</div>
      
      <div class="short-answer-section">
        <div class="section-label">Short Answer</div>
        <div class="short-answer-text">${escapeHtml(shortAnswer)}</div>
      </div>
      
      <div class="reasoning-section">
        <div class="section-label">Reasoning</div>
        <div class="reasoning-text">${escapeHtml(reasoning)}</div>
      </div>
      
      ${sourcesHtml}
    `;
    
    conversationArea.appendChild(botDiv);
    scrollToBottom();
  }

  function addLoadingIndicator() {
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "bot-response loading-indicator";
    loadingDiv.id = "loading-indicator";
    loadingDiv.innerHTML = `
      <div class="bot-label">BOT RESPONSE</div>
      <div class="short-answer-text">Processing your question...</div>
    `;
    conversationArea.appendChild(loadingDiv);
    scrollToBottom();
    return loadingDiv;
  }

  function removeLoadingIndicator() {
    const loading = document.getElementById("loading-indicator");
    if (loading) {
      loading.remove();
    }
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  async function sendMessage() {
    if (sending) return;

    const text = userInput.value.trim();
    if (!text) return;

    sending = true;
    sendBtn.disabled = true;

    // Remove welcome message on first question
    if (welcomeMessage) {
      welcomeMessage.remove();
    }

    // Add user question
    addUserMessage(text);
    userInput.value = "";

    // Show loading
    const loadingIndicator = addLoadingIndicator();

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: text
        })
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      sessionId = data.session_id;

      // Remove loading indicator
      removeLoadingIndicator();

      // Parse response
      const reply = data.reply || "";
      const parts = reply.split("REASONING:");
      
      let shortAnswer = "No answer provided.";
      let reasoning = "";
      
      if (parts.length >= 2) {
        shortAnswer = parts[0].replace("SHORT ANSWER:", "").trim();
        reasoning = parts[1].trim();
      } else if (parts.length === 1) {
        shortAnswer = reply.replace("SHORT ANSWER:", "").trim();
      }

      // Add bot response
      addBotResponse(shortAnswer, reasoning, data.sources || []);

    } catch (err) {
      console.error("Error:", err);
      removeLoadingIndicator();
      
      addBotResponse(
        "Error contacting server.",
        "There was a problem connecting to the rules database. Please try again.",
        []
      );
    } finally {
      sending = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  }

  // Button click
  sendBtn.addEventListener("click", sendMessage);

  // Enter key (without Shift)
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-focus input
  userInput.focus();
});