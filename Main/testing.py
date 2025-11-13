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
PROMPTS_FILE = "prompts.json"

if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY environment variable not set. Set it before running.")
else:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="📊 AI Data Analytics Engine", layout="wide")

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
    """Generate a generic summary that works for any dataset"""
    insights = [f"📊 **Dataset Overview**: {len(df):,} rows, {df.shape[1]} columns"]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        insights.append(f"🔢 **Numeric Columns**: {len(numeric_cols)}")
    
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        insights.append(f"📝 **Categorical Columns**: {len(cat_cols)}")
    
    missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 0:
        insights.append(f"⚠️ **Missing Data**: {missing_pct:.1f}%")
    
    if len(df.columns) > 0:
        first_col = df.columns[0]
        if df[first_col].dtype == 'object':
            unique_count = df[first_col].nunique()
            insights.append(f"🎯 **'{first_col}' has {unique_count} unique values**")
    
    return "\n\n".join(insights)

def plot_histogram(df, col):
    return px.histogram(df, x=col, marginal="box", nbins=40, title=f"Distribution of {col}")

def plot_scatter(df, x, y, color=None):
    return px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}", height=500)

def plot_bar(df, x, y, agg='mean'):
    agg_df = df.groupby(x)[y].agg(agg).reset_index()
    return px.bar(agg_df, x=x, y=y, title=f"{agg.capitalize()} of {y} by {x}", height=500)

def plot_line(df, x, y):
    return px.line(df, x=x, y=y, title=f"{y} over {x}", markers=True, height=500)

def plot_pie(df, col):
    pie_data = df[col].value_counts().head(10).reset_index()
    pie_data.columns = [col, "count"]
    return px.pie(pie_data, names=col, values="count", title=f"Distribution of {col}", height=500)

def plot_correlation_heatmap(df):
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty or len(df_num.columns) < 2:
        return None
    corr = df_num.corr()
    fig = px.imshow(corr, text_auto='.2f', aspect="auto", title="Correlation Matrix")
    return fig

def generate_dynamic_system_prompt(df: pd.DataFrame) -> str:
    """Generate a smart system prompt based on the uploaded dataset"""
    
    cols_info = []
    for col in df.columns[:20]:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = f"(mean: {df[col].mean():.2f}, range: {df[col].min():.1f}-{df[col].max():.1f})"
            cols_info.append(f"- {col}: numeric {stats}")
        else:
            unique = df[col].nunique()
            sample_vals = df[col].dropna().astype(str).value_counts().head(3).index.tolist()
            cols_info.append(f"- {col}: categorical ({unique} unique values, e.g., {', '.join(sample_vals[:3])})")
    
    columns_desc = "\n".join(cols_info)
    
    prompt = f"""You are an expert data analyst assistant helping users analyze their dataset.

DATASET INFORMATION:
- Total rows: {len(df):,}
- Total columns: {len(df.columns)}

COLUMNS:
{columns_desc}

YOUR CAPABILITIES:
1. **Data Filtering & Queries**: Answer questions about the data, filter rows based on conditions
2. **Visualizations**: Create charts automatically when users request plots
3. **Statistical Analysis**: Calculate means, correlations, distributions, trends
4. **General Knowledge**: Answer domain-related questions even if not directly in the data
5. **Data Insights**: Identify patterns, anomalies, and provide actionable insights

CRITICAL INSTRUCTION FOR PLOTS:
When the user asks to plot, graph, chart, show, visualize, or display ANY relationship between columns, you MUST respond with a PLOT command.

**For ANY PLOT/GRAPH/CHART request, you MUST include:**
PLOT: {{"type": "scatter|bar|line|pie|histogram", "x": "exact_column_name", "y": "exact_column_name"}}

Plot type selection:
- Use "scatter" for: "plot X vs Y", "show relationship", "X against Y"
- Use "bar" for: "bar chart", "compare by category"
- Use "line" for: "over time", "trend", "progression"
- Use "histogram" for: "distribution of", "spread of"
- Use "pie" for: "pie chart", "breakdown of", "proportion"

MANDATORY EXAMPLES:
User: "Plot Machine ID vs Units Produced"
Response: "I'll create a scatter plot to show the relationship between Machine ID and Units Produced.

PLOT: {{"type": "scatter", "x": "Machine ID", "y": "Units Produced"}}"

User: "Show me a graph of sales by region"
Response: "I'll create a bar chart showing sales by region.

PLOT: {{"type": "bar", "x": "region", "y": "sales"}}"

User: "Display age distribution"
Response: "I'll create a histogram to show the distribution of ages.

PLOT: {{"type": "histogram", "x": "age"}}"

**For DATA FILTERING:**
QUERY: {{"column_name": "value"}} for exact/partial match
QUERY: {{"column_name": {{"$gt": value}}}} for greater than
QUERY: {{"column_name": {{"$lt": value}}}} for less than

IMPORTANT RULES:
1. ALWAYS include PLOT: command when user requests visualization
2. Use EXACT column names from the dataset (case-sensitive)
3. Never say you can't display graphs - always provide the PLOT: command
4. Keep natural language response brief, focus on the PLOT: command
5. If unsure which columns, ask for clarification but still suggest a PLOT: command"""

    return prompt

def load_or_generate_prompt(df: pd.DataFrame = None) -> str:
    """Load custom prompt from file, or generate dynamic one from dataset"""
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            custom_prompt = data.get("system_prompt", "")
            if custom_prompt.strip():
                return custom_prompt
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"⚠️ Error reading {PROMPTS_FILE}: {e}")
    
    if df is not None:
        return generate_dynamic_system_prompt(df)
    
    return "You are an expert data analyst assistant."

def save_prompt_to_json(prompt: str, file_path: str = PROMPTS_FILE):
    """Save the updated prompt to JSON file."""
    try:
        data = {"system_prompt": prompt}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving prompt: {e}")
        return False

def call_openai_chat(system: str, user_prompt: str, model: str = OPENAI_MODEL, max_tokens: int = 1000) -> str:
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

def detect_plot_intent(user_question: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Fallback: Detect plot intent from user question using keyword matching"""
    question_lower = user_question.lower()
    
    # Keywords that indicate plot request
    plot_keywords = ['plot', 'graph', 'chart', 'show', 'visualize', 'display', 'draw']
    is_plot_request = any(keyword in question_lower for keyword in plot_keywords)
    
    if not is_plot_request:
        return None
    
    # Extract column names from question
    mentioned_cols = []
    for col in df.columns:
        if col.lower() in question_lower:
            mentioned_cols.append(col)
    
    if len(mentioned_cols) >= 2:
        # Determine plot type based on keywords
        if 'bar' in question_lower:
            return {"type": "bar", "x": mentioned_cols[0], "y": mentioned_cols[1]}
        elif 'line' in question_lower:
            return {"type": "line", "x": mentioned_cols[0], "y": mentioned_cols[1]}
        elif 'pie' in question_lower:
            return {"type": "pie", "x": mentioned_cols[0]}
        else:
            # Default to scatter for "vs" or general plot
            return {"type": "scatter", "x": mentioned_cols[0], "y": mentioned_cols[1]}
    
    elif len(mentioned_cols) == 1:
        # Single column - default to histogram
        return {"type": "histogram", "x": mentioned_cols[0]}
    
    return None

def parse_query_from_response(response: str) -> Optional[Dict[str, Any]]:
    if not response or "QUERY:" not in response:
        return None
    try:
        query_part = response.split("QUERY:")[1].strip()
        json_match = re.search(r'\{.*?\}', query_part, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        return None
    return None

def find_best_column_match(user_col: str, available_cols: List[str]) -> Optional[str]:
    """Find the best matching column name (case-insensitive, fuzzy)"""
    user_col_lower = user_col.lower().strip()
    
    # Exact match (case-insensitive)
    for col in available_cols:
        if col.lower() == user_col_lower:
            return col
    
    # Partial match
    for col in available_cols:
        if user_col_lower in col.lower() or col.lower() in user_col_lower:
            return col
    
    # Check for common variations
    user_col_normalized = user_col_lower.replace('_', ' ').replace('-', ' ')
    for col in available_cols:
        col_normalized = col.lower().replace('_', ' ').replace('-', ' ')
        if user_col_normalized == col_normalized:
            return col
    
    return None

def create_plot_from_spec(df: pd.DataFrame, plot_spec: Dict[str, Any]):
    """Create a plot based on AI-generated specifications"""
    try:
        plot_type = plot_spec.get('type', '').lower()
        x_col = plot_spec.get('x')
        y_col = plot_spec.get('y')
        color_col = plot_spec.get('color')
        
        # Find matching columns
        x_matched = find_best_column_match(x_col, df.columns.tolist()) if x_col else None
        y_matched = find_best_column_match(y_col, df.columns.tolist()) if y_col else None
        color_matched = find_best_column_match(color_col, df.columns.tolist()) if color_col else None
        
        if plot_type == 'scatter' and x_matched and y_matched:
            return px.scatter(df, x=x_matched, y=y_matched, color=color_matched, 
                            title=f"{y_matched} vs {x_matched}", height=500)
        
        elif plot_type == 'bar' and x_matched and y_matched:
            return plot_bar(df, x_matched, y_matched)
        
        elif plot_type == 'line' and x_matched and y_matched:
            return plot_line(df, x_matched, y_matched)
        
        elif plot_type == 'histogram' and x_matched:
            return plot_histogram(df, x_matched)
        
        elif plot_type == 'pie' and x_matched:
            return plot_pie(df, x_matched)
        
        else:
            return None
            
    except Exception as e:
        st.error(f"Error creating plot: {e}")
        return None

def parse_plot_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract plot specifications (PLOT: {...}) from AI response text."""
    if not response or "PLOT:" not in response:
        return None
    try:
        # Extract the JSON part following PLOT:
        plot_part = response.split("PLOT:")[1].strip()
        json_match = re.search(r'\{.*\}', plot_part, re.DOTALL)
        if json_match:
            plot_json = json_match.group()
            return json.loads(plot_json)
    except Exception:
        return None
    return None

# Main UI
st.title("📊 AI Data Analytics Engine")
st.markdown("Upload any CSV file to explore, visualize, and chat with your data using AI.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_sample = st.number_input("Rows to show in sample", value=5, min_value=1, step=1)
    show_heatmap = st.checkbox("Show correlation heatmap", value=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])
    
    st.markdown("---")
    
    with st.expander("🤖 AI Prompt Editor (Advanced)", expanded=False):
        st.markdown("**Edit the system prompt** or let AI auto-generate based on your data:")
        
        st.info("💡 Leave blank to auto-generate smart prompts based on your data!")
        
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                current_custom_prompt = data.get("system_prompt", "")
        except:
            current_custom_prompt = ""
        
        edited_prompt = st.text_area(
            "Custom System Prompt (optional)",
            value=current_custom_prompt,
            height=250,
            help="Leave empty to auto-generate intelligent prompts based on your dataset",
            key="prompt_editor",
            placeholder="Leave blank for auto-generated prompts..."
        )
        
        col_save, col_clear = st.columns(2)
        
        with col_save:
            if st.button("💾 Save Custom", type="primary", use_container_width=True):
                if save_prompt_to_json(edited_prompt):
                    st.success("✅ Saved!")
                    st.rerun()
        
        with col_clear:
            if st.button("🔄 Clear (Use Auto)", use_container_width=True):
                if save_prompt_to_json(""):
                    st.success("✅ Will auto-generate!")
                    st.rerun()

# Load data
if uploaded_file:
    df = load_csv(uploaded_file)
    st.success(f"✅ Loaded: {uploaded_file.name}")
else:
    if os.path.exists(DEFAULT_CSV_PATH):
        df = pd.read_csv(DEFAULT_CSV_PATH, low_memory=False)
        st.info(f"📂 Using example dataset: {DEFAULT_CSV_PATH}")
    else:
        st.warning("👆 Please upload a CSV file to begin analysis")
        st.stop()

# Dataset Summary
st.subheader("📊 Dataset Overview")
st.markdown(generate_intelligent_summary(df))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("Total Columns", df.shape[1])
with col3:
    st.metric("Missing Values", int(df.isna().sum().sum()))

st.dataframe(df.head(max_sample), use_container_width=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Visualizations
st.markdown("---")
st.subheader("📈 Interactive Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Histogram", "🔵 Scatter", "📉 Bar", "🥧 Pie"])

with tab1:
    if numeric_cols:
        col = st.selectbox("Select numeric column", numeric_cols)
        st.plotly_chart(plot_histogram(df, col), use_container_width=True)
    else:
        st.info("No numeric columns available for histogram.")

with tab2:
    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols, key="xaxis")
        y = st.selectbox("Y-axis", numeric_cols, key="yaxis")
        color = st.selectbox("Color by (optional)", [None] + cat_cols, index=0)
        st.plotly_chart(plot_scatter(df, x, y, color if color else None), use_container_width=True)
    else:
        st.info("Need at least two numeric columns for scatter plot.")

with tab3:
    if cat_cols and numeric_cols:
        x = st.selectbox("Categorical column", cat_cols, key="barx")
        y = st.selectbox("Numeric column", numeric_cols, key="bary")
        agg = st.selectbox("Aggregation", ["sum", "mean", "count"], index=1)
        agg_df = df.groupby(x)[y].agg(agg).reset_index()
        st.plotly_chart(px.bar(agg_df, x=x, y=y, title=f"{agg.capitalize()} of {y} by {x}"), use_container_width=True)
    else:
        st.info("Need both categorical and numeric columns for bar chart.")

with tab4:
    if cat_cols:
        col = st.selectbox("Select categorical column", cat_cols, key="pie")
        pie_data = df[col].value_counts().head(10).reset_index()
        pie_data.columns = [col, "count"]
        st.plotly_chart(px.pie(pie_data, names=col, values="count", title=f"Distribution of {col}"), use_container_width=True)
    else:
        st.info("No categorical columns available for pie chart.")

if show_heatmap:
    fig = plot_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Chat Interface
st.markdown("---")
st.subheader("💬 Chat with Your Data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Generate dynamic examples based on data
example_questions = []
if len(numeric_cols) >= 2:
    example_questions.append(f"Plot {numeric_cols[0]} vs {numeric_cols[1]}")
if numeric_cols:
    example_questions.append(f"Show records where {numeric_cols[0]} > average")
if cat_cols:
    example_questions.append(f"Show distribution of {cat_cols[0]}")

examples_text = " • ".join(example_questions[:3]) if example_questions else "Ask anything about your data!"

user_question = st.text_input(
    f"💭 Ask anything about your data:",
    key="chat",
    placeholder=f"e.g., {examples_text}"
)

if st.button("🚀 Send", type="primary") and user_question:
    st.session_state.chat_history.append(("user", user_question))
    
    # Get or generate system prompt
    sys_prompt = load_or_generate_prompt(df)
    
    # Build context
    context_text = f"Dataset: {len(df):,} rows, {len(df.columns)} columns.\nColumns: {', '.join(df.columns)}"
    
    user_prompt = f"{context_text}\n\nUser question: {user_question}"
    
    response = call_openai_chat(sys_prompt, user_prompt)
    st.session_state.chat_history.append(("assistant", response))
    
    # Try to extract plot from AI response first
    plot_spec = parse_plot_from_response(response)
    
    # If AI didn't provide PLOT command, try fallback detection
    if not plot_spec:
        plot_spec = detect_plot_intent(user_question, df)
        if plot_spec:
            st.session_state.chat_history.append(("assistant", "📊 Creating visualization..."))
    
    # Generate the plot if spec exists
    if plot_spec:
        fig = create_plot_from_spec(df, plot_spec)
        if fig:
            st.session_state.chat_history.append(("plot", fig))
        else:
            st.session_state.chat_history.append(("assistant", "⚠️ Could not create the requested plot. Please check column names."))
    
    # Check for data query
    query_dict = parse_query_from_response(response)
    if query_dict:
        try:
            filtered = df.copy()
            for col, condition in query_dict.items():
                matched_col = find_best_column_match(col, df.columns.tolist())
                if matched_col:
                    if isinstance(condition, dict):
                        op = list(condition.keys())[0]
                        val = condition[op]
                        if op == "$gt":
                            filtered = filtered[pd.to_numeric(filtered[matched_col], errors='coerce') > val]
                        elif op == "$lt":
                            filtered = filtered[pd.to_numeric(filtered[matched_col], errors='coerce') < val]
                        elif op == "$gte":
                            filtered = filtered[pd.to_numeric(filtered[matched_col], errors='coerce') >= val]
                        elif op == "$lte":
                            filtered = filtered[pd.to_numeric(filtered[matched_col], errors='coerce') <= val]
                        elif op == "$eq":
                            filtered = filtered[filtered[matched_col].astype(str).str.lower() == str(val).lower()]
                    else:
                        filtered = filtered[filtered[matched_col].astype(str).str.contains(str(condition), case=False, na=False)]
            
            if len(filtered) > 0:
                st.session_state.chat_history.append(("dataframe", filtered))
            else:
                st.session_state.chat_history.append(("assistant", "⚠️ No records match the filter criteria."))
        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"⚠️ Could not filter data: {e}"))

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation History")
    for item in st.session_state.chat_history:
        if item[0] == "user":
            st.markdown(f"**👤 You:** {item[1]}")
        elif item[0] == "assistant":
            st.markdown(f"**🤖 AI:** {item[1]}")
        elif item[0] == "dataframe":
            st.dataframe(item[1], use_container_width=True)
            st.caption(f"📊 Showing {len(item[1])} matching records")
        elif item[0] == "plot":
            # Give every chart a unique key, even if identical
            unique_key = f"plot_{hash(str(item[1]))}_{len(st.session_state.chat_history)}"
            st.plotly_chart(item[1], use_container_width=True, key=unique_key)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Try asking 'Plot X vs Y' or 'Show me a graph of...' to create visualizations!")