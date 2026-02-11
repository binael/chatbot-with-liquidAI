async function sendMessage() {
  const input = document.getElementById("user-input");
  const text = input.value;

  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });

  const data = await response.json();

  addMessage(data.response, "bot");
}

function addMessage(text, type) {
  const box = document.getElementById("chat-box");

  const msg = document.createElement("div");
  msg.className = `message ${type}`;
  msg.innerText = text;

  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
}
