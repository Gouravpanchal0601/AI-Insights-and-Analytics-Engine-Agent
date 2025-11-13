import os
import io
import json
import re
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_CSV_PATH = "doctor.csv"
PROMPTS_FILE = "new.json"

if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY environment variable not set. Set it before running.")
else:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="🩺 AI Insights & Analytics Engine", layout="wide")

@st.cache_data
def load_csv(file_bytes: io.BytesIO) -> pd.DataFrame:
    try:
        file_bytes.seek(0)
        df = pd.read_csv(file_bytes)
        return df
    except Exception:
        file_bytes.seek(0)
        df = pd.read_csv(file_bytes, encoding='latin1', low_memory=False)
        return df

def nullable_to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def dataset_brief(df: pd.DataFrame, n_sample: int = 5) -> Dict[str, Any]:
    brief = {'n_rows': len(df), 'n_cols': len(df.columns), 'columns': []}
    for col in df.columns:
        col_info = {"name": str(col), "dtype": str(df[col].dtype), "n_missing": int(df[col].isna().sum())}
        if pd.api.types.is_numeric_dtype(df[col]):
            nonnull = df[col].dropna()
            if not nonnull.empty:
                col_info.update({
                    'mean': float(nonnull.mean()),
                    'median': float(nonnull.median()),
                    'std': float(nonnull.std()),
                    'min': float(nonnull.min()),
                    'max': float(nonnull.max()),
                })
        else:
            nonnull = df[col].dropna().astype(str)
            col_info.update({
                'n_unique': int(nonnull.nunique()),
                'top_values': list(nonnull.value_counts().head(5).index.astype(str))
            })
        brief['columns'].append(col_info)
    brief['sample_rows'] = df.head(n_sample).to_dict(orient='records')
    return brief

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
    return px.histogram(df, x=col, marginal="box", nbins=40, title=f"Histogram of {col}")

def plot_scatter(df, x, y, color=None):
    return px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")

def plot_correlation_heatmap(df):
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty:
        return None
    corr = df_num.corr()
    fig = px.imshow(corr, text_auto='.2f', aspect="auto", title="Correlation Matrix")
    return fig

def load_prompt_from_json(file_path: str = PROMPTS_FILE, key: str = "system_prompt") -> str:
    """Load the system prompt text from an external JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(key, "You are an expert data analyst assistant.")
    except FileNotFoundError:
        st.warning(f"⚠️ {file_path} not found. Using default prompt.")
        return "You are an expert data analyst assistant."
    except Exception as e:
        st.warning(f"⚠️ Error reading {file_path}: {e}")
        return "You are an expert data analyst assistant."

def save_prompt_to_json(prompt: str, file_path: str = PROMPTS_FILE, key: str = "system_prompt"):
    """Save the updated prompt to JSON file."""
    try:
        # Load existing data
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = {}
        
        # Update prompt
        data[key] = prompt
        
        # Save back to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving prompt: {e}")
        return False

def call_openai_chat(system: str, user_prompt: str, model: str = OPENAI_MODEL, max_tokens: int = 700) -> str:
    if not openai.api_key:
        return "⚠️ Missing API key."
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

def parse_query_from_response(response: str) -> Optional[Dict[str, Any]]:
    if not response or "QUERY:" not in response:
        return None
    try:
        query_part = response.split("QUERY:")[1].strip()
        json_match = re.search(r'\{.*\}', query_part, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        return None
    return None

st.title("🩺 AI Insights & Analytics Engine")
st.markdown("Upload your CSV to explore, visualize, and chat with your data using AI.")
st.markdown("You are a much smarter ai agent ")

with st.sidebar:
    st.header("⚙️ Settings")
    max_sample = st.number_input("Rows to show in sample", value=5, min_value=1, step=1)
    show_heatmap = st.checkbox("Show correlation heatmap", value=True)
    
    st.markdown("---")
    
    with st.expander("🤖 AI Prompt Editor", expanded=False):
        st.markdown("Edit the system prompt used by AI assistant:")
        
        current_prompt = load_prompt_from_json()
        
        edited_prompt = st.text_area(
            "System Prompt",
            value=current_prompt,
            height=300,
            help="This prompt guides the AI's behavior and responses",
            key="prompt_editor"
        )
        
        col_save, col_reset = st.columns(2)
        
        with col_save:
            if st.button("💾 Save Prompt", type="primary", use_container_width=True):
                if save_prompt_to_json(edited_prompt):
                    st.success("✅ Prompt saved!")
                    st.rerun()
        
        with col_reset:
            if st.button("🔄 Load from File", use_container_width=True):
                st.rerun()
    
    st.markdown("---")
    st.markdown("💡 **Tips:**")
    st.markdown("- 'Show all diabetes patients'")
    st.markdown("- 'List patients with high severity'")
    st.markdown("- 'Find heart disease cases'")
    st.markdown("- 'Plot cost vs age'")
    st.markdown("- 'How to control sugar' (general)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = load_csv(uploaded_file)
else:
    df = pd.read_csv(DEFAULT_CSV_PATH, low_memory=False)
    st.info(f"📂 Loaded default dataset: {DEFAULT_CSV_PATH}")

st.subheader("📊 Dataset Summary")
st.markdown(generate_intelligent_summary(df))
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Missing Values", int(df.isna().sum().sum()))

st.dataframe(df.head(max_sample), use_container_width=True)
brief = dataset_brief(df, n_sample=max_sample)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

st.markdown("---")
st.subheader("📈 Interactive Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Histogram", "🔵 Scatter", "📉 Bar", "🥧 Pie"])
with tab1:
    if numeric_cols:
        col = st.selectbox("Select numeric column", numeric_cols)
        st.plotly_chart(plot_histogram(df, col), use_container_width=True)
    else:
        st.info("No numeric columns available.")

with tab2:
    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols, key="xaxis")
        y = st.selectbox("Y-axis", numeric_cols, key="yaxis")
        color = st.selectbox("Color (optional)", [None] + cat_cols, index=0)
        st.plotly_chart(plot_scatter(df, x, y, color if color else None), use_container_width=True)
    else:
        st.info("Need at least two numeric columns.")

with tab3:
    if cat_cols and numeric_cols:
        x = st.selectbox("Categorical", cat_cols, key="barx")
        y = st.selectbox("Numeric", numeric_cols, key="bary")
        agg = st.selectbox("Aggregation", ["sum", "mean", "count"], index=1)
        agg_df = df.groupby(x)[y].agg(agg).reset_index()
        st.plotly_chart(px.bar(agg_df, x=x, y=y, title=f"{agg.capitalize()} of {y} by {x}"), use_container_width=True)
    else:
        st.info("Need categorical and numeric columns.")

with tab4:
    if cat_cols:
        col = st.selectbox("Select categorical column", cat_cols, key="pie")
        pie_data = df[col].value_counts().head(10).reset_index()
        pie_data.columns = [col, "count"]
        st.plotly_chart(px.pie(pie_data, names=col, values="count", title=f"Distribution of {col}"), use_container_width=True)
    else:
        st.info("No categorical columns available.")

if show_heatmap:
    fig = plot_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("💬 Chat with Your Data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

context_text = f"Dataset: {len(df)} rows, {len(df.columns)} columns.\nColumns: {', '.join(df.columns)}"
user_question = st.text_input("Ask anything (e.g., 'Show all patients above 60', 'Plot age vs bill')", key="chat")

if st.button("🚀 Send", type="primary") and user_question:
    st.session_state.chat_history.append(("user", user_question))

    sys_prompt = load_prompt_from_json()  # Load from JSON
    user_prompt = f"Dataset Context:\n{context_text}\n\nUser question: {user_question}"

    response = call_openai_chat(sys_prompt, user_prompt)
    st.session_state.chat_history.append(("assistant", response))

    # Try extracting query
    query_dict = parse_query_from_response(response)
    if query_dict:
        try:
            filtered = df.copy()
            for col, condition in query_dict.items():
                if col in filtered.columns:
                    if isinstance(condition, dict):
                        op = list(condition.keys())[0]
                        val = condition[op]
                        if op == "$gt":
                            filtered = filtered[pd.to_numeric(filtered[col], errors='coerce') > val]
                        elif op == "$lt":
                            filtered = filtered[pd.to_numeric(filtered[col], errors='coerce') < val]
                    else:
                        filtered = filtered[filtered[col].astype(str).str.contains(str(condition), case=False)]
            st.session_state.chat_history.append(("dataframe", filtered))
        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"⚠️ Could not filter data: {e}"))

if st.session_state.chat_history:
    st.markdown("### 🧾 Conversation History")
    for item in st.session_state.chat_history:
        if item[0] == "user":
            st.markdown(f"**👤 You:** {item[1]}")
        elif item[0] == "assistant":
            st.markdown(f"**🤖 Assistant:** {item[1]}")
        elif item[0] == "dataframe":
            st.dataframe(item[1], use_container_width=True)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()