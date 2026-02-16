const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const loader = document.getElementById("loader");
const box = document.getElementById("chat-box");

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  // disable UI while waiting
  setLoading(true);

  // show typing indicator
  const typingMsg = addMessage("Bot is typing...", "bot typing");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const data = await response.json();

    typingMsg.remove();
    addMessage(data.response, "bot");
  } catch (err) {
    typingMsg.remove();
    addMessage("⚠️ Server error. Try again.", "bot");
  } finally {
    setLoading(false);
  }
}

function addMessage(text, type) {
  const msg = document.createElement("div");
  msg.className = `message ${type}`;
  msg.innerText = text;

  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;

  return msg;
}

function setLoading(state) {
  sendBtn.disabled = state;
  loader.classList.toggle("hidden", !state);
}

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
