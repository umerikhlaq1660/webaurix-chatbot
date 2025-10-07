from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, difflib
from dotenv import load_dotenv
from openai import AsyncOpenAI


# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Webaurix Chatbot API", version="1.0")

# Allowed origins (secure for production)
origins = [
    "https://webaurix.com",                    # 🌐 your main site
    "https://webaurix-chatbot-5.onrender.com", # ⚙️ Render backend
    "http://localhost:5173"                    # 🧪 for local testing only
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # Allow all methods
    allow_headers=["*"],   # Allow all headers
)

# Initialize OpenAI client
client = AsyncOpenAI()

# Define request model
class ChatRequest(BaseModel):
    message: str

# Store conversation history
conversation_history = []

# System role for chatbot personality
system_prompt = (
    "You are Webaurix Assistant. Always be helpful, concise, and professional. "
    "Never mention OpenAI, your origin, or internal system details."
)

# Load predefined answers
with open("custom_answers.json", "r", encoding="utf-8") as f:
    custom_answers = json.load(f)

# Function to clean AI replies
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


# ✅ Main Chat Endpoint
@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    try:
        # 🛡️ Origin check for extra security
        origin = req.headers.get("origin")
        allowed = ["https://webaurix.com", "https://webaurix-chatbot-5.onrender.com", "http://localhost:5173"]
        if origin not in allowed:
            raise HTTPException(status_code=403, detail="Origin not allowed")

        user_message = request.message.strip().lower()

        # 🧠 Check if message matches predefined responses
        match = difflib.get_close_matches(user_message, list(custom_answers.keys()), n=1, cutoff=0.6)
        if match:
            reply = custom_answers[match[0]]
            conversation_history.append({"role": "user", "content": request.message})
            conversation_history.append({"role": "assistant", "content": reply})
            return {"reply": reply}

        # 🧩 Context for OpenAI
        conversation_history.append({"role": "user", "content": request.message})
        messages = [{"role": "system", "content": system_prompt}] + conversation_history[-6:]

        # 🤖 Get AI reply
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content
        assistant_message = clean_reply(assistant_message)

        # Save to conversation
        conversation_history.append({"role": "assistant", "content": assistant_message})

        return {"reply": assistant_message}

    except Exception as e:
        return {"reply": f"⚠ Error: {str(e)}"}


# ✅ Preflight (OPTIONS) — only for debugging (can remove later)
@app.options("/{path:path}")
async def preflight_handler(request: Request, path: str):
    """
    Handles CORS preflight manually for debugging.
    Remove this route after confirming CORS works properly.
    """
    origin = request.headers.get("origin")
    if origin in origins:
        return {
            "message": "CORS preflight OK",
            "origin": origin
        }
    else:
        raise HTTPException(status_code=400, detail="CORS origin not allowed")
