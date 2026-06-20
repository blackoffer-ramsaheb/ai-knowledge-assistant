const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const messages = document.getElementById("messages");
const thinking = document.getElementById("thinking");
const sendButton = form.querySelector("button");

function addMessage(text, role) {
	const message = document.createElement("div");
	message.className = `message ${role}`;
	message.textContent = text;
	messages.appendChild(message);
	messages.scrollTop = messages.scrollHeight;
}

function setThinking(isThinking) {
	thinking.classList.toggle("visible", isThinking);
	thinking.setAttribute("aria-hidden", String(!isThinking));
	input.disabled = isThinking;
	sendButton.disabled = isThinking;
}

form.addEventListener("submit", async (event) => {
	event.preventDefault();

	const userMessage = input.value.trim();
	if (!userMessage) {
		return;
	}

	addMessage(userMessage, "user");
	input.value = "";
	setThinking(true);

	try {
		const response = await fetch("/chat", {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({ message: userMessage })
		});

		const data = await response.json();
		addMessage(data.response || "No response returned.", "bot");
	} catch (error) {
		addMessage("Unable to reach the chatbot backend.", "bot");
	} finally {
		setThinking(false);
	}
});
