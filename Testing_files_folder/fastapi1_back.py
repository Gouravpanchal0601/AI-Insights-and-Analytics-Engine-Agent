import os
import io
import json
import re
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import plotly.express as px

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROMPTS_FILE = "new.json"
DEFAULT_CSV_PATH = "doctor.csv"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment.")
openai.api_key = OPENAI_API_KEY

app = FastAPI(title="🩺 AI Insights & Analytics Engine", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_csv(file_bytes: io.BytesIO) -> pd.DataFrame:
    try:
        file_bytes.seek(0)
        return pd.read_csv(file_bytes)
    except Exception:
        file_bytes.seek(0)
        return pd.read_csv(file_bytes, encoding='latin1', low_memory=False)

def load_prompt_from_json(file_path: str = PROMPTS_FILE, key: str = "system_prompt") -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(key, "You are an expert data analyst assistant.")
    except Exception:
        return "You are an expert data analyst assistant."

def save_prompt_to_json(prompt: str, file_path: str = PROMPTS_FILE, key: str = "system_prompt"):
    data = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[key] = prompt
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def call_openai_chat(system: str, user_prompt: str, model: str = OPENAI_MODEL, max_tokens: int = 700) -> str:
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI request failed: {e}"

def generate_intelligent_summary(df: pd.DataFrame) -> str:
    insights = [f"📊 **Dataset Overview**: {len(df)} rows, {df.shape[1]} columns"]
    if 'hospital_branch' in df.columns:
        insights[-1] += f" across {df['hospital_branch'].nunique()} branches"
    if 'readmitted' in df.columns:
        try:
            rate = (df['readmitted'].astype(str).str.lower() == 'yes').mean() * 100
            insights.append(f"🔁 **Readmission Rate**: {rate:.1f}%")
        except:
            pass
    if 'total_bill' in df.columns:
        try:
            avg_bill = df['total_bill'].mean()
            insights.append(f"💰 **Avg Bill**: ₹{avg_bill:,.0f}")
        except:
            pass
    if 'satisfaction_score' in df.columns:
        avg_sat = df['satisfaction_score'].mean()
        insights.append(f"⭐ **Avg Satisfaction**: {avg_sat:.1f}/10")
    if 'diagnosis' in df.columns:
        top = df['diagnosis'].value_counts().head(3)
        insights.append(f"🥼 **Top Diagnoses**: {', '.join(top.index)}")
    return "\n\n".join(insights)

def plot_histogram(df, col):
    return px.histogram(df, x=col, marginal="box", nbins=40, title=f"Histogram of {col}").to_json()

def plot_scatter(df, x, y, color=None):
    return px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}").to_json()

def plot_correlation_heatmap(df):
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty:
        return None
    return px.imshow(df_num.corr(), text_auto='.2f', aspect="auto", title="Correlation Matrix").to_json()

DATASET: Optional[pd.DataFrame] = None

class ChatRequest(BaseModel):
    question: str

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "🩺 AI Insights & Analytics Engine API is running!"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global DATASET
    contents = await file.read()
    DATASET = load_csv(io.BytesIO(contents))
    return {
        "filename": file.filename,
        "rows": len(DATASET),
        "columns": list(DATASET.columns),
        "message": "✅ CSV uploaded successfully!"
    }

@app.get("/summary")
def get_summary():
    if DATASET is None:
        return {"error": "No dataset loaded. Upload a CSV first."}
    return {"summary": generate_intelligent_summary(DATASET)}
    
@app.get("/visualize/histogram")
def visualize_histogram(column: str):
    if DATASET is None:
        return {"error": "No dataset loaded."}
    if column not in DATASET.columns:
        return {"error": f"Column '{column}' not found."}
    return json.loads(plot_histogram(DATASET, column))

@app.get("/visualize/scatter")
def visualize_scatter(x: str, y: str, color: Optional[str] = None):
    if DATASET is None:
        return {"error": "No dataset loaded."}
    return json.loads(plot_scatter(DATASET, x, y, color))

@app.get("/visualize/heatmap")
def visualize_heatmap():
    if DATASET is None:
        return {"error": "No dataset loaded."}
    fig = plot_correlation_heatmap(DATASET)
    return json.loads(fig) if fig else {"error": "No numeric columns to plot."}

@app.post("/chat")
def chat_with_data(req: ChatRequest):
    if DATASET is None:
        return {"error": "No dataset loaded."}

    sys_prompt = load_prompt_from_json()
    context = f"Dataset: {len(DATASET)} rows, {len(DATASET.columns)} columns.\nColumns: {', '.join(DATASET.columns)}"
    user_prompt = f"Dataset Context:\n{context}\n\nUser question: {req.question}"

    response = call_openai_chat(sys_prompt, user_prompt)
    return {"response": response}

@app.get("/prompt")
def get_prompt():
    prompt = load_prompt_from_json()
    return {"prompt": prompt}

@app.post("/prompt")
def save_prompt(req: PromptRequest):
    save_prompt_to_json(req.prompt)
    return {"message": "✅ Prompt saved successfully!"}
