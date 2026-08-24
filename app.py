

import os
import re
import sys
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DATA_FILE = os.getenv("DATA_FILE", "data.txt")
TOP_K = int(os.getenv("TOP_K", "3"))  # number of chunks to retrieve

if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found. Add it to your .env file.")
    sys.exit(1)


# ----------------------------------------------------------------------
# STEP 1: Load & chunk the knowledge base
# ----------------------------------------------------------------------
def load_chunks(path: str):
    """
    Splits data.txt into chunks using [SECTION HEADERS] as boundaries.
    Falls back to paragraph splitting if no headers are found.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on lines like [SOMETHING]
    parts = re.split(r"(?=\[[A-Z0-9 &-]+\])", text)
    chunks = [p.strip() for p in parts if p.strip()]

    if not chunks:
        # fallback: split on blank lines
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

    return chunks


# ----------------------------------------------------------------------
# STEP 2: Build a local TF-IDF retriever (no external API required)
# ----------------------------------------------------------------------
class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = TOP_K):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = [self.chunks[i] for i in top_idx if sims[i] > 0]
        # If nothing scored above 0 (very short/odd query), just return top_k anyway
        if not results:
            results = [self.chunks[i] for i in top_idx]
        return results


# ----------------------------------------------------------------------
# STEP 3: Call OpenRouter for the final answer
# ----------------------------------------------------------------------
def ask_openrouter(question: str, context_chunks: list):
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are the official AI assistant for Greenfield Private School. "
        "Answer the user's question using ONLY the CONTEXT provided below. "
        "If the answer is not present in the context, politely say you don't "
        "have that information and suggest contacting the school office. "
        "Be concise, friendly, and accurate. Do not make up facts.\n\n"
        f"CONTEXT:\n{context}"
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for analytics/rate-limit tiers:
        "HTTP-Referer": "https://greenfieldprivateschool.edu",
        "X-Title": "Greenfield School RAG Assistant",
    }

    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        return f"[Error {resp.status_code}] {resp.text}"

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return f"[Unexpected response format]: {data}"


# ----------------------------------------------------------------------
# STEP 4: RAG pipeline
# ----------------------------------------------------------------------
def rag_answer(question: str, retriever: Retriever):
    context_chunks = retriever.retrieve(question)
    answer = ask_openrouter(question, context_chunks)
    return answer, context_chunks


# ----------------------------------------------------------------------
# CLI MODE (default)
# ----------------------------------------------------------------------
def run_cli():
    print("Loading knowledge base...")
    chunks = load_chunks(DATA_FILE)
    retriever = Retriever(chunks)
    print(f"Loaded {len(chunks)} chunks from {DATA_FILE}")
    print("Greenfield Private School RAG Assistant (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        answer, used_chunks = rag_answer(question, retriever)
        print(f"\nAssistant: {answer}\n")


# ----------------------------------------------------------------------
# OPTIONAL: WEB MODE (Flask) — set RUN_MODE=web in .env to use this
# ----------------------------------------------------------------------
def run_web():
    from flask import Flask, request, jsonify, render_template_string

    app = Flask(__name__)
    chunks = load_chunks(DATA_FILE)
    retriever = Retriever(chunks)

    PAGE = """
    <!doctype html>
    <html>
    <head>
      <title>Greenfield School Assistant</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; }
        #chat { border: 1px solid #ccc; border-radius: 8px; padding: 15px; height: 400px; overflow-y: auto; }
        .msg { margin: 8px 0; }
        .user { color: #1a73e8; font-weight: bold; }
        .bot { color: #333; }
        #q { width: 80%; padding: 8px; }
        button { padding: 8px 16px; }
      </style>
    </head>
    <body>
      <h2>🏫 Greenfield Private School — AI Assistant</h2>
      <div id="chat"></div>
      <br>
      <input id="q" placeholder="Ask about admissions, fees, timings..." onkeydown="if(event.key==='Enter')send()">
      <button onclick="send()">Send</button>
      <script>
        async function send() {
          const q = document.getElementById('q').value;
          if (!q) return;
          const chat = document.getElementById('chat');
          chat.innerHTML += `<div class="msg user">You: ${q}</div>`;
          document.getElementById('q').value = '';
          chat.innerHTML += `<div class="msg bot" id="pending">Assistant: thinking...</div>`;
          chat.scrollTop = chat.scrollHeight;
          const res = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: q})
          });
          const data = await res.json();
          document.getElementById('pending').outerHTML = `<div class="msg bot">Assistant: ${data.answer}</div>`;
          chat.scrollTop = chat.scrollHeight;
        }
      </script>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    @app.route("/ask", methods=["POST"])
    def ask():
        question = request.json.get("question", "")
        answer, _ = rag_answer(question, retriever)
        return jsonify({"answer": answer})

    app.run(host="0.0.0.0", port=5000, debug=False)


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    mode = os.getenv("RUN_MODE", "cli").lower()
    if mode == "web":
        run_web()
    else:
        run_cli()
