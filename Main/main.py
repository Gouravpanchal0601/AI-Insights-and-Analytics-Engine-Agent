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
import re
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
    return px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")

def plot_correlation_heatmap(df):
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty or len(df_num.columns) < 2:
        return None
    corr = df_num.corr()
    fig = px.imshow(corr, text_auto='.2f', aspect="auto", title="Correlation Matrix")
    return fig

def generate_dynamic_system_prompt(df: pd.DataFrame) -> str:
    """Generate a smart system prompt based on the uploaded dataset"""
    
    # Get dataset characteristics
    cols_info = []
    for col in df.columns[:20]:  # Limit to first 20 columns for prompt brevity
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
2. **Visualizations**: Suggest and create charts (histograms, scatter plots, bar charts, pie charts)
3. **Statistical Analysis**: Calculate means, correlations, distributions, trends
4. **General Knowledge**: Answer domain-related questions even if not directly in the data
5. **Data Insights**: Identify patterns, anomalies, and provide actionable insights

RESPONSE FORMAT:
- For data queries: Provide clear natural language response, then add QUERY: followed by a JSON filter
- For visualizations: Suggest chart types and columns to use
- For general questions: Provide informative answers based on your knowledge
- Always be helpful even if the user's question is vague or poorly worded

QUERY JSON FORMAT (when filtering data):
QUERY: {{"column_name": "value"}} for exact/partial match
QUERY: {{"column_name": {{"$gt": value}}}} for greater than
QUERY: {{"column_name": {{"$lt": value}}}} for less than

EXAMPLES:
- "Show me records where X is above 100" → Natural answer + QUERY: {{"X": {{"$gt": 100}}}}
- "What does Y mean?" → Explain Y conceptually using your knowledge
- "Plot A vs B" → "I recommend a scatter plot of A vs B to see the relationship"
- "Find all Z containing 'keyword'" → Natural answer + QUERY: {{"Z": "keyword"}}

IMPORTANT:
- Understand user intent even with unclear phrasing
- Be conversational and helpful
- Provide insights beyond just filtering data
- Suggest next steps or additional analyses when relevant"""

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
    
    # Generate dynamic prompt if no custom one exists
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

def compute_math_query(df, question: str) -> Optional[str]:
    """Detect and compute simple math operations like avg, sum, max, min."""
    q = question.lower()

    operations = {
        "average": "mean",
        "avg": "mean",
        "mean": "mean",
        "sum": "sum",
        "total": "sum",
        "maximum": "max",
        "max": "max",
        "minimum": "min",
        "min": "min",
        "largest": "max",
        "smallest": "min",
        "count": "count",
        "how many": "count",
        "number of": "count"
    }

    # Try to find operation
    op = None
    for word, func in operations.items():
        if word in q:
            op = func
            break

    if not op:
        return None

    # Try to find matching column name
    matched_col = None
    for col in df.columns:
        clean_col = col.lower().replace("_", "").replace(" ", "")
        clean_q = q.replace("_", "").replace(" ", "")
        if clean_col in clean_q:
            matched_col = col
            break

    if not matched_col:
        return None

    try:
        series = pd.to_numeric(df[matched_col], errors="coerce")
        result = None
        if op == "mean":
            result = series.mean()
        elif op == "sum":
            result = series.sum()
        elif op == "max":
            result = series.max()
        elif op == "min":
            result = series.min()
        elif op == "count":
            result = series.count()

        if pd.isna(result):
            return None
        return f"🧮 The **{op}** of `{matched_col}` is **{result:.2f}**."
    except Exception as e:
        return f"⚠️ Could not compute {op} for {matched_col}: {e}"

st.title("📊 AI Data Analytics Engine")
st.markdown("Upload any CSV file to explore, visualize, and chat with your data using AI.")

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

st.markdown("---")
st.subheader("💬 Chat with Your Data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input(
    "💭 Ask anything about your data:",
    placeholder="e.g., Plot Machine ID vs Units Produced, show correlation between cost and score"
)

if st.button("🚀 Send", type="primary") and user_question:
    st.session_state.chat_history.append(("user", user_question))

    # ✅ Try to compute mathematical queries directly from the dataset
    math_result = compute_math_query(df, user_question)
    if math_result:
        st.session_state.chat_history.append(("assistant", math_result))
    else:
        # Fallback to AI if not a simple math question
        sys_prompt = load_or_generate_prompt(df)
        context_text = f"Dataset: {len(df)} rows, {len(df.columns)} columns.\nColumns: {', '.join(df.columns)}"
        user_prompt = f"{context_text}\n\nUser question: {user_question}"
        response = call_openai_chat(sys_prompt, user_prompt)
        st.session_state.chat_history.append(("assistant", response))

    sys_prompt = load_or_generate_prompt(df)
    context_text = f"Dataset: {len(df)} rows, {len(df.columns)} columns.\nColumns: {', '.join(df.columns)}"
    user_prompt = f"{context_text}\n\nUser question: {user_question}"

    response = call_openai_chat(sys_prompt, user_prompt)
    st.session_state.chat_history.append(("assistant", response))

    # --- Detect and Auto-Plot ---
    plot_match = re.search(r"[Pp]lot\s+([\w\s]+)\s+vs\s+([\w\s]+)", user_question)
    if not plot_match:
        plot_match = re.search(r"[Pp]lot\s+([\w\s]+)\s+against\s+([\w\s]+)", user_question)

    if plot_match:
        x_col = plot_match.group(1).strip()
        y_col = plot_match.group(2).strip()

        # Helper: match column names case-insensitively
        def match_column(name):
            for col in df.columns:
                clean = lambda s: s.lower().replace(" ", "").replace("_", "")
                if clean(col) == clean(name):
                    return col
            return None

        x_real = match_column(x_col)
        y_real = match_column(y_col)

        if x_real and y_real:
            st.success(f"📊 Auto-plotting **{x_real} vs {y_real}**")
            fig = px.scatter(df, x=x_real, y=y_real, title=f"{y_real} vs {x_real}")
            st.plotly_chart(fig, use_container_width=True)

            # ✅ Store plot in chat history so it persists
            st.session_state.chat_history.append(("plot", (x_real, y_real, fig)))
        else:
            st.warning(f"⚠️ Couldn't match '{x_col}' or '{y_col}' to dataset columns.")

    # --- Handle JSON Query ---
    query_dict = parse_query_from_response(response)
# ---------- Enhanced query execution with normalization, nested eval, and semantic checks ----------
    if query_dict:
        try:
            filtered = df.copy()

            # Normalize operator names
            def normalize_op(op_raw: str) -> str:
                if not isinstance(op_raw, str):
                    return ""
                op = op_raw.lower()
                if op.startswith("$"):
                    op = op[1:]
                return op  # e.g., "gt", "lt", "eq", "gte", "lte"

            def compute_value_for_column(ref_col: str, agg: str):
                """Compute aggregate (mean,max,min) of ref_col safely."""
                if ref_col not in filtered.columns:
                    return None
                # coerce to numeric where possible
                ser = pd.to_numeric(filtered[ref_col], errors="coerce")
                if agg in ("avg", "mean"):
                    return ser.mean()
                if agg in ("max", "maximum"):
                    return ser.max()
                if agg in ("min", "minimum"):
                    return ser.min()
                if agg in ("sum", "total"):
                    return ser.sum()
                return None

            def looks_like_identifier(colname: str, series: pd.Series) -> bool:
                """Heuristic: columns containing 'id' or mostly integers with few unique values -> likely identifier"""
                name_lower = colname.lower()
                if "id" in name_lower or "code" in name_lower:
                    return True
                # if dtype non-numeric and many unique values -> id-like
                if not pd.api.types.is_numeric_dtype(series):
                    return True
                # numeric but integer-like with many unique? if unique fraction high maybe not id; if low it's id
                uniq_frac = series.dropna().nunique() / max(1, len(series))
                if pd.api.types.is_integer_dtype(series) and uniq_frac < 0.05:
                    return True
                return False

            def compute_dynamic_value(col, raw_val):
                """Interpret values like number, 'average', or {'avg':'$Col'}"""
                # direct numeric
                if isinstance(raw_val, (int, float)):
                    return float(raw_val)

                # string like "average" or numeric-as-string
                if isinstance(raw_val, str):
                    rv = raw_val.strip().lower()
                    if rv in ("average", "mean"):
                        return pd.to_numeric(filtered[col], errors="coerce").mean()
                    if rv in ("max", "maximum"):
                        return pd.to_numeric(filtered[col], errors="coerce").max()
                    if rv in ("min", "minimum"):
                        return pd.to_numeric(filtered[col], errors="coerce").min()
                    # if it's a number in string form
                    asnum = pd.to_numeric(raw_val, errors="coerce")
                    if not pd.isna(asnum):
                        return float(asnum)
                    return None

                # dict like {"avg":"$Machine ID"} or {"avg": "$Other"}
                if isinstance(raw_val, dict):
                    # expect a single key like {"avg": "$Col"}
                    for k, v in raw_val.items():
                        agg = k.lower()
                        # if value is a string referencing column with '$'
                        if isinstance(v, str) and v.startswith("$"):
                            ref_col = v[1:].strip()
                            if ref_col in filtered.columns:
                                # semantic check: if ref_col looks like id, warn and fallback (handled above)
                                return compute_value_for_column(ref_col, agg)
                        # if v is a literal number, return it
                        if isinstance(v, (int, float)):
                            return float(v)
                    return None

                return None

            # Main filter application
            warnings_msgs = []
            for col, condition in query_dict.items():
                if col not in filtered.columns:
                    warnings_msgs.append(f"Column '{col}' not found in dataset.")
                    continue

                # If condition is a dict, expect an operator -> value
                if isinstance(condition, dict):
                    op_raw = list(condition.keys())[0]
                    raw_val = condition[op_raw]
                    op = normalize_op(op_raw)

                    dyn_val = compute_dynamic_value(col, raw_val)

                    # If dyn_val is None and raw_val is a dict with a $ref, try to compute from ref even if different column
                    if dyn_val is None and isinstance(raw_val, dict):
                        # try computing from any referenced column inside raw_val
                        for v in raw_val.values():
                            if isinstance(v, str) and v.startswith("$"):
                                ref = v[1:].strip()
                                # sanity: if ref exists compute mean (or respective agg)
                                if ref in filtered.columns:
                                    # if comparing to an id-like column, fallback to same-col mean instead
                                    if looks_like_identifier(ref, filtered[ref]):
                                        fallback_mean = pd.to_numeric(filtered[col], errors="coerce").mean()
                                        warnings_msgs.append(
                                            f"Ref column '{ref}' looks like an identifier — comparing to its aggregate is likely meaningless. "
                                            f"Falling back to mean of '{col}' ({fallback_mean:.2f})."
                                        )
                                        dyn_val = fallback_mean
                                    else:
                                        # compute requested agg of ref
                                        # get agg name
                                        agg_key = list(raw_val.keys())[0]
                                        dyn_val = compute_value_for_column(ref, agg_key)
                                        break

                    if dyn_val is None:
                        warnings_msgs.append(f"Could not resolve comparison value for `{col}` with raw `{raw_val}`.")
                        # skip applying this filter
                        continue

                    # Apply operator (elementwise). Use to_numeric on column for safe comparison
                    left_ser = pd.to_numeric(filtered[col], errors="coerce")
                    if op in ("gt",):
                        filtered = filtered[left_ser > dyn_val]
                    elif op in ("lt",):
                        filtered = filtered[left_ser < dyn_val]
                    elif op in ("gte",):
                        filtered = filtered[left_ser >= dyn_val]
                    elif op in ("lte",):
                        filtered = filtered[left_ser <= dyn_val]
                    elif op in ("eq",):
                        filtered = filtered[left_ser == dyn_val]
                    else:
                        warnings_msgs.append(f"Unsupported operator '{op_raw}' for column '{col}'.")
                else:
                    # simple contains / equality
                    filtered = filtered[filtered[col].astype(str).str.contains(str(condition), case=False, na=False)]

            # Show warnings as assistant messages (if any)
            for w in warnings_msgs:
                st.session_state.chat_history.append(("assistant", f"⚠️ {w}"))

            if len(filtered) > 0:
                st.session_state.chat_history.append(("dataframe", filtered))
            else:
                st.session_state.chat_history.append(("assistant", "⚠️ No records match the filter criteria."))

        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"⚠️ Could not filter data: {e}"))
            try:
                filtered = df.copy()

                def compute_dynamic_value(value):
                    """Helper to interpret 'average', 'max', or {"avg": "$col"} structures."""
                    if isinstance(value, (int, float)):
                        return value
                    if isinstance(value, str):
                        if value.lower() in ["average", "mean"]:
                            return filtered[col].mean()
                        elif value.lower() in ["max", "maximum"]:
                            return filtered[col].max()
                        elif value.lower() in ["min", "minimum"]:
                            return filtered[col].min()
                        else:
                            return pd.to_numeric(value, errors="coerce")
                    if isinstance(value, dict):
                        # handle {"avg": "$Machine ID"} or {"mean": "$Col"}
                        for k, v in value.items():
                            if isinstance(v, str) and v.startswith("$"):
                                ref_col = v[1:]
                                if ref_col in filtered.columns:
                                    if k.lower() in ["avg", "mean"]:
                                        return filtered[ref_col].mean()
                                    elif k.lower() in ["max", "maximum"]:
                                        return filtered[ref_col].max()
                                    elif k.lower() in ["min", "minimum"]:
                                        return filtered[ref_col].min()
                            elif isinstance(v, (int, float)):
                                return v
                    return None

                for col, condition in query_dict.items():
                    if col in filtered.columns:
                        if isinstance(condition, dict):
                            op = list(condition.keys())[0]
                            val = condition[op]

                            # dynamically compute nested values like {"avg": "$Machine ID"}
                            dyn_val = compute_dynamic_value(val)

                            if pd.isna(dyn_val):
                                continue

                            if op in ["$gt", "gt"]:
                                filtered = filtered[pd.to_numeric(filtered[col], errors='coerce') > dyn_val]
                            elif op in ["$lt", "lt"]:
                                filtered = filtered[pd.to_numeric(filtered[col], errors='coerce') < dyn_val]
                            elif op in ["$gte", "gte"]:
                                filtered = filtered[pd.to_numeric(filtered[col], errors='coerce') >= dyn_val]
                            elif op in ["$lte", "lte"]:
                                filtered = filtered[pd.to_numeric(filtered[col], errors='coerce') <= dyn_val]
                            elif op in ["$eq", "eq"]:
                                filtered = filtered[filtered[col].astype(str).str.lower() == str(dyn_val).lower()]
                        else:
                            filtered = filtered[filtered[col].astype(str).str.contains(str(condition), case=False, na=False)]

                if len(filtered) > 0:
                    st.session_state.chat_history.append(("dataframe", filtered))
                else:
                    st.session_state.chat_history.append(("assistant", "⚠️ No records match the filter criteria."))

            except Exception as e:
                st.session_state.chat_history.append(("assistant", f"⚠️ Could not filter data: {e}"))

if st.session_state.chat_history:
    st.markdown("### 💬 Conversation History")
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**👤 You:** {content}")
        elif role == "assistant":
            st.markdown(f"**🤖 AI:** {content}")
        elif role == "dataframe":
            st.dataframe(content, use_container_width=True)
            st.caption(f"📊 Showing {len(content)} matching records")
        elif role == "plot":
            x_real, y_real, fig = content
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"📈 Persistent plot: {y_real} vs {x_real}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()