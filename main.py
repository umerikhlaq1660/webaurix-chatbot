from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json
from dotenv import load_dotenv
from openai import AsyncOpenAI
import difflib

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://webaurix.com/",
        "http://127.0.0.1:https://webaurix.com/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI()

class ChatRequest(BaseModel):
    message: str

conversation_history = []
system_prompt = "You are Webaurix Assistant. Always be helpful, concise, and professional. Never mention OpenAI or your origin."


with open("custom_answers.json", "r", encoding="utf-8") as f:
    custom_answers = json.load(f)


def clean_reply(text: str) -> str:
    blocked_phrases = [
        "I was developed by OpenAI",
        "an artificial intelligence research organization",
        "as an AI developed by OpenAI",
        "I am a language model created by OpenAI"
    ]
    for phrase in blocked_phrases:
        if phrase.lower() in text.lower():
            return "I’m your Webaurix Assistant, here to help you!"
    return text

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_message = request.message.strip().lower()

        
        keys = list(custom_answers.keys())
        match = difflib.get_close_matches(user_message, keys, n=1, cutoff=0.6)

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
            max_tokens=500 
        )

        assistant_message = response.choices[0].message.content

        
        assistant_message = clean_reply(assistant_message)

        conversation_history.append({"role": "assistant", "content": assistant_message})

        return {"reply": assistant_message}
    
    except Exception as e:
        return {"reply": f"⚠ Error: {str(e)}"}

