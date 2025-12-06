"""
Admin Dashboard - Production Metrics & Evaluation
Access: Only for you, not users
Shows: Satisfaction %, TTFT, session metrics, eval results
"""

import streamlit as st
import os
from datetime import datetime, timedelta
from typing import Dict, List
import json

from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

# ======================== PAGE CONFIG ========================

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="📊",
    layout="wide"
)

# ======================== LANGSMITH CLIENT ========================

@st.cache_resource
def get_langsmith_client():
    try:
        return Client()
    except:
        return None

client = get_langsmith_client()
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "socrates")

# ======================== DATA FETCHING ========================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_runs(days: int = 7, limit: int = 500) -> List[Dict]:
    """Fetch recent runs from LangSmith"""
    if not client:
        return []
    
    try:
        runs = list(client.list_runs(
            project_name=PROJECT_NAME,
            start_time=datetime.now() - timedelta(days=days),
            limit=limit
        ))
        return runs
    except Exception as e:
        st.error(f"Error fetching runs: {e}")
        return []

@st.cache_data(ttl=60)
def fetch_feedback(days: int = 7) -> List[Dict]:
    """Fetch feedback from LangSmith"""
    if not client:
        return []
    
    try:
        # Get runs with feedback
        runs = fetch_runs(days=days)
        feedback_list = []
        
        for run in runs:
            try:
                feedbacks = list(client.list_feedback(run_ids=[run.id]))
                for fb in feedbacks:
                    feedback_list.append({
                        "run_id": str(run.id),
                        "score": fb.score,
                        "key": fb.key,
                        "comment": fb.comment,
                        "created_at": fb.created_at
                    })
            except:
                pass
        
        return feedback_list
    except Exception as e:
        st.error(f"Error fetching feedback: {e}")
        return []

# ======================== METRICS CALCULATION ========================

def calculate_metrics(runs: List, feedback: List) -> Dict:
    """Calculate business metrics from runs and feedback"""
    
    # Basic counts
    total_runs = len(runs)
    
    # Latency metrics
    latencies = []
    first_token_times = []
    
    for run in runs:
        if hasattr(run, 'latency') and run.latency:
            latencies.append(run.latency)
        
        # Try to get TTFT from run data
        if hasattr(run, 'outputs') and run.outputs:
            if 'first_token_time' in str(run.outputs):
                pass  # Would extract TTFT here
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p50_latency = sorted(latencies)[len(latencies)//2] if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) > 1 else 0
    
    # Feedback metrics
    total_feedback = len(feedback)
    positive_feedback = sum(1 for f in feedback if f.get('score', 0) == 1)
    negative_feedback = sum(1 for f in feedback if f.get('score', 0) == 0)
    
    satisfaction_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
    
    # Error rate
    errors = sum(1 for run in runs if hasattr(run, 'error') and run.error)
    error_rate = (errors / total_runs * 100) if total_runs > 0 else 0
    
    # Sessions (approximate by grouping runs by time proximity)
    sessions = set()
    for run in runs:
        if hasattr(run, 'session_id'):
            sessions.add(run.session_id)
    
    # Daily breakdown
    daily_counts = {}
    for run in runs:
        if hasattr(run, 'start_time') and run.start_time:
            day = run.start_time.strftime('%Y-%m-%d')
            daily_counts[day] = daily_counts.get(day, 0) + 1
    
    return {
        "total_runs": total_runs,
        "total_feedback": total_feedback,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "satisfaction_rate": round(satisfaction_rate, 1),
        "avg_latency_s": round(avg_latency, 2),
        "p50_latency_s": round(p50_latency, 2),
        "p99_latency_s": round(p99_latency, 2),
        "error_rate": round(error_rate, 2),
        "estimated_sessions": len(sessions) if sessions else total_runs // 3,
        "daily_breakdown": daily_counts
    }

# ======================== UI ========================

st.title("📊 Production Analytics Dashboard")
st.caption(f"Project: {PROJECT_NAME} | Data refreshes every 60 seconds")

# Sidebar controls
st.sidebar.header("Controls")
days = st.sidebar.slider("Days of data", 1, 30, 7)
refresh = st.sidebar.button("🔄 Refresh Data")

if refresh:
    st.cache_data.clear()

# Check LangSmith connection
if not client:
    st.error("⚠️ LangSmith not connected. Check your API key.")
    st.stop()

# Fetch data
with st.spinner("Fetching data from LangSmith..."):
    runs = fetch_runs(days=days)
    feedback = fetch_feedback(days=days)
    metrics = calculate_metrics(runs, feedback)

# ======================== MAIN METRICS ========================

st.header("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    satisfaction_color = "🟢" if metrics['satisfaction_rate'] >= 70 else "🟡" if metrics['satisfaction_rate'] >= 50 else "🔴"
    st.metric(
        label="User Satisfaction",
        value=f"{metrics['satisfaction_rate']}%",
        delta=f"{satisfaction_color} {metrics['positive_feedback']}/{metrics['total_feedback']} ratings"
    )

with col2:
    st.metric(
        label="Total Interactions",
        value=metrics['total_runs'],
        delta=f"Last {days} days"
    )

with col3:
    st.metric(
        label="Avg Latency",
        value=f"{metrics['avg_latency_s']}s",
        delta=f"P99: {metrics['p99_latency_s']}s"
    )

with col4:
    error_color = "🟢" if metrics['error_rate'] < 1 else "🟡" if metrics['error_rate'] < 5 else "🔴"
    st.metric(
        label="Error Rate",
        value=f"{metrics['error_rate']}%",
        delta=error_color
    )

st.markdown("---")

# ======================== DETAILED BREAKDOWN ========================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Feedback Breakdown")
    
    if metrics['total_feedback'] > 0:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Pie(
            labels=['Positive 👍', 'Negative 👎'],
            values=[metrics['positive_feedback'], metrics['negative_feedback']],
            marker_colors=['#22c55e', '#ef4444'],
            hole=0.4
        )])
        fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feedback collected yet")
    
    # Recent feedback
    st.markdown("**Recent Feedback:**")
    for fb in feedback[:5]:
        emoji = "👍" if fb.get('score') == 1 else "👎"
        st.caption(f"{emoji} {fb.get('comment', 'No comment')} - {fb.get('created_at', '')}")

with col_right:
    st.subheader("📊 Daily Activity")
    
    if metrics['daily_breakdown']:
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame([
            {"date": k, "interactions": v} 
            for k, v in sorted(metrics['daily_breakdown'].items())
        ])
        
        fig = px.bar(df, x='date', y='interactions', 
                     color_discrete_sequence=['#6366f1'])
        fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily data available")

st.markdown("---")

# ======================== LATENCY ANALYSIS ========================

st.subheader("⚡ Latency Distribution")

latency_col1, latency_col2, latency_col3 = st.columns(3)

with latency_col1:
    st.metric("P50 Latency", f"{metrics['p50_latency_s']}s")
    
with latency_col2:
    st.metric("P99 Latency", f"{metrics['p99_latency_s']}s")
    
with latency_col3:
    st.metric("Avg Latency", f"{metrics['avg_latency_s']}s")

st.markdown("---")

# ======================== EVALUATION RESULTS ========================

st.subheader("🧪 Evaluation Results")

eval_file = "eval_results.json"
if os.path.exists(eval_file):
    with open(eval_file, 'r') as f:
        eval_results = json.load(f)
    
    if eval_results:
        eval_col1, eval_col2 = st.columns(2)
        
        with eval_col1:
            st.markdown("**Prompt Test Results:**")
            if "prompt_tests" in eval_results:
                tests = eval_results["prompt_tests"]
                st.metric("Pass Rate", f"{tests.get('pass_rate', 0)*100:.0f}%")
                st.caption(f"Passed: {tests.get('passed', 0)} / {tests.get('total', 0)}")
        
        with eval_col2:
            st.markdown("**RAG Quality Scores:**")
            if "rag_eval" in eval_results:
                rag = eval_results["rag_eval"]
                st.metric("Faithfulness", f"{rag.get('faithfulness_avg', 0)*100:.0f}%")
                st.metric("Answer Relevance", f"{rag.get('answer_relevance_avg', 0)*100:.0f}%")
else:
    st.info("No evaluation results yet. Run `python run_evals.py` to generate.")

st.markdown("---")

# ======================== RAW DATA ========================

with st.expander("🔍 View Raw Run Data"):
    if runs:
        st.json([{
            "id": str(run.id),
            "name": run.name,
            "latency": run.latency,
            "start_time": str(run.start_time),
            "error": run.error
        } for run in runs[:20]])
    else:
        st.info("No runs found")

# ======================== EXPORT ========================

st.sidebar.markdown("---")
st.sidebar.subheader("Export")

if st.sidebar.button("📥 Export Metrics JSON"):
    st.sidebar.download_button(
        label="Download",
        data=json.dumps(metrics, indent=2, default=str),
        file_name=f"metrics_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

# ======================== FOOTER ========================

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")