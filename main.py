from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json, difflib, os


load_dotenv()

app = FastAPI(title="Webaurix Chatbot API", version="1.1")


origins = [
    "https://webaurix.com",
    "https://webaurix-chatbot-5.onrender.com",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],     
    allow_headers=["*"],     
)

client = AsyncOpenAI()


class ChatRequest(BaseModel):
    message: str



conversation_history = []
system_prompt = (
    "You are Webaurix Assistant. Always be helpful, concise, and professional. "
    "Never mention OpenAI, your origin, or internal system details."
)


with open("custom_answers.json", "r", encoding="utf-8") as f:
    custom_answers = json.load(f)



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

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_message = request.message.strip().lower()

        
        match = difflib.get_close_matches(user_message, list(custom_answers.keys()), n=1, cutoff=0.6)
        if match:
            reply = custom_answers[match[0]]
            conversation_history.append({"role": "user", "content": request.message})
            conversation_history.append({"role": "assistant", "content": reply})
            return {"reply": reply}

      
        conversation_history.append({"role": "user", "content": request.message})
        messages = [{"role": "system", "content": system_prompt}] + conversation_history[-6:]

     
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=1500
        )

        assistant_message = clean_reply(response.choices[0].message.content)
        conversation_history.append({"role": "assistant", "content": assistant_message})

        return {"reply": assistant_message}

    except Exception as e:
        print("❌ Error:", str(e))
        return {"reply": f"⚠ Error: {str(e)}"}
