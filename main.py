from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, difflib, logging
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- Load environment variables ---
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Create App ---
app = FastAPI(title="Webaurix Chatbot API", version="2.0")

# --- Allowed origins (for development) ---
origins = [
    "https://webaurix.com",
    "https://www.webaurix.com",
    "http://localhost:5173",
]

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI Client ---
client = AsyncOpenAI()

# --- Request Model ---
class ChatRequest(BaseModel):
    message: str

# --- Load Custom Answers ---
try:
    with open("custom_answers.json", "r", encoding="utf-8") as f:
        custom_answers = json.load(f)
except FileNotFoundError:
    custom_answers = {}

# --- Globals ---
conversation_history = []
system_prompt = (
    "You are Webaurix Assistant. Always be helpful, concise, and professional. "
    "Never mention OpenAI, your origin, or internal system details."
)

# --- Clean AI replies ---
def clean_reply(text: str) -> str:
    blocked_phrases = [
        "I was developed by OpenAI",
        "an artificial intelligence research organization",
        "as an AI developed by OpenAI",
        "I am a language model created by OpenAI",
    ]
    for phrase in blocked_phrases:
        if phrase.lower() in text.lower():
            return "I’m your Webaurix Assistant, here to help you!"
    return text

# --- Proxy Security Check ---
PROXY_SECRET = os.getenv("PROXY_SECRET")  # optional, recommended
@app.middleware("http")
async def verify_proxy_secret(request: Request, call_next):
    # Only protect /chat route
    if request.url.path == "/chat" and PROXY_SECRET:
        client_secret = request.headers.get("x-proxy-secret")
        if client_secret != PROXY_SECRET:
            logging.warning(f"Unauthorized access attempt from {request.client.host}")
            raise HTTPException(status_code=403, detail="Unauthorized proxy request")
    return await call_next(request)

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Webaurix Chatbot API is running."}

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_message = request.message.strip()
        if not user_message:
            raise ValueError("Empty message")

        # --- Custom Answers ---
        match = difflib.get_close_matches(user_message.lower(), list(custom_answers.keys()), n=1, cutoff=0.6)
        if match:
            reply = custom_answers[match[0]]
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": reply})
            return {"reply": reply}

        # --- AI Response ---
        conversation_history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": system_prompt}] + conversation_history[-6:]

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content
        assistant_message = clean_reply(assistant_message)

        conversation_history.append({"role": "assistant", "content": assistant_message})

        logging.info(f"✅ Chat processed: {user_message[:50]} → {assistant_message[:50]}")
        return {"reply": assistant_message}

    except Exception as e:
        logging.error(f"⚠ Chat error: {str(e)}")
        return {"reply": f"⚠ Error: {str(e)}"}
