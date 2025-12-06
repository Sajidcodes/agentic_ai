"""
Analytics module for RAG chatbot
Tracks: satisfaction, session metrics, response times, retention
"""

import streamlit as st
from datetime import datetime, timedelta
import json
import os
import time
from pathlib import Path

# ======================== CONFIGURATION ========================
ANALYTICS_FILE = "analytics_data.json"

# ======================== DATA STRUCTURES ========================

def get_default_analytics():
    """Default structure for analytics data"""
    return {
        "sessions": [],
        "interactions": [],
        "feedback": [],
        "daily_stats": {}
    }

def get_default_session():
    """Default structure for a session"""
    return {
        "session_id": None,
        "start_time": None,
        "end_time": None,
        "mode": None,  # "rag", "socratic", "chat"
        "interaction_count": 0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        "queries": []
    }

# ======================== PERSISTENCE ========================

def load_analytics():
    """Load analytics from file"""
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, 'r') as f:
                return json.load(f)
        except:
            return get_default_analytics()
    return get_default_analytics()

def save_analytics(data):
    """Save analytics to file"""
    try:
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving analytics: {e}")

# ======================== SESSION TRACKING ========================

def init_session_analytics():
    """Initialize analytics for current session"""
    if 'analytics_session' not in st.session_state:
        st.session_state['analytics_session'] = {
            "session_id": st.session_state.get('thread_id', 'unknown'),
            "start_time": datetime.now().isoformat(),
            "interactions": [],
            "feedback": {"up": 0, "down": 0},
            "mode_usage": {"rag": 0, "socratic": 0, "chat": 0}
        }
    
    if 'ttft_start' not in st.session_state:
        st.session_state['ttft_start'] = None

def start_response_timer():
    """Call this when user submits a query"""
    st.session_state['ttft_start'] = time.time()

def get_ttft():
    """Get time-to-first-token in milliseconds"""
    if st.session_state.get('ttft_start'):
        ttft = (time.time() - st.session_state['ttft_start']) * 1000
        st.session_state['ttft_start'] = None
        return round(ttft, 2)
    return None

# ======================== INTERACTION TRACKING ========================

def log_interaction(query: str, response: str, mode: str, ttft_ms: float = None):
    """Log a single interaction"""
    init_session_analytics()
    
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "query": query[:500],  # Truncate long queries
        "response_length": len(response),
        "mode": mode,
        "ttft_ms": ttft_ms
    }
    
    st.session_state['analytics_session']['interactions'].append(interaction)
    st.session_state['analytics_session']['mode_usage'][mode] = \
        st.session_state['analytics_session']['mode_usage'].get(mode, 0) + 1
    
    # Persist to file
    analytics = load_analytics()
    analytics['interactions'].append({
        **interaction,
        "session_id": st.session_state['analytics_session']['session_id']
    })
    save_analytics(analytics)

def log_feedback(is_positive: bool, query: str = None):
    """Log thumbs up/down feedback"""
    init_session_analytics()
    
    feedback_type = "up" if is_positive else "down"
    st.session_state['analytics_session']['feedback'][feedback_type] += 1
    
    # Persist
    analytics = load_analytics()
    analytics['feedback'].append({
        "timestamp": datetime.now().isoformat(),
        "session_id": st.session_state['analytics_session']['session_id'],
        "is_positive": is_positive,
        "query": query[:200] if query else None
    })
    save_analytics(analytics)

# ======================== METRICS CALCULATION ========================

def calculate_metrics():
    """Calculate all UX metrics from stored data"""
    analytics = load_analytics()
    
    # Satisfaction Score
    total_feedback = len(analytics['feedback'])
    positive_feedback = sum(1 for f in analytics['feedback'] if f['is_positive'])
    satisfaction_score = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
    
    # Average TTFT
    ttfts = [i['ttft_ms'] for i in analytics['interactions'] if i.get('ttft_ms')]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    
    # Session metrics
    sessions = {}
    for interaction in analytics['interactions']:
        sid = interaction.get('session_id', 'unknown')
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(interaction)
    
    # Average interactions per session
    avg_session_length = sum(len(s) for s in sessions.values()) / len(sessions) if sessions else 0
    
    # Mode distribution
    mode_counts = {"rag": 0, "socratic": 0, "chat": 0}
    for interaction in analytics['interactions']:
        mode = interaction.get('mode', 'chat')
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    total_interactions = len(analytics['interactions'])
    
    return {
        "satisfaction_score": round(satisfaction_score, 1),
        "total_feedback": total_feedback,
        "positive_feedback": positive_feedback,
        "negative_feedback": total_feedback - positive_feedback,
        "avg_ttft_ms": round(avg_ttft, 2),
        "total_sessions": len(sessions),
        "total_interactions": total_interactions,
        "avg_session_length": round(avg_session_length, 1),
        "mode_distribution": mode_counts,
        "interactions_today": sum(
            1 for i in analytics['interactions'] 
            if i['timestamp'][:10] == datetime.now().strftime('%Y-%m-%d')
        )
    }

# ======================== STREAMLIT UI COMPONENTS ========================

def render_feedback_buttons(query: str, response_index: int):
    """Render thumbs up/down buttons for a response"""
    col1, col2, col3 = st.columns([1, 1, 10])
    
    feedback_key = f"feedback_{response_index}"
    
    # Check if already voted
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = None
    
    with col1:
        if st.button("👍", key=f"up_{response_index}", 
                     disabled=st.session_state[feedback_key] is not None):
            log_feedback(True, query)
            st.session_state[feedback_key] = "up"
            st.rerun()
    
    with col2:
        if st.button("👎", key=f"down_{response_index}",
                     disabled=st.session_state[feedback_key] is not None):
            log_feedback(False, query)
            st.session_state[feedback_key] = "down"
            st.rerun()
    
    # Show confirmation
    if st.session_state[feedback_key]:
        with col3:
            st.caption("Thanks for your feedback!")

def render_metrics_dashboard():
    """Render analytics dashboard in sidebar or main area"""
    metrics = calculate_metrics()
    
    st.markdown("### 📊 Analytics Dashboard")
    
    # Top metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        satisfaction_color = "🟢" if metrics['satisfaction_score'] >= 70 else "🟡" if metrics['satisfaction_score'] >= 50 else "🔴"
        st.metric(
            label="Satisfaction Score",
            value=f"{metrics['satisfaction_score']}%",
            delta=f"{satisfaction_color}"
        )
    
    with col2:
        st.metric(
            label="Avg TTFT",
            value=f"{metrics['avg_ttft_ms']}ms"
        )
    
    with col3:
        st.metric(
            label="Sessions",
            value=metrics['total_sessions']
        )
    
    # Second row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric(
            label="Total Interactions",
            value=metrics['total_interactions']
        )
    
    with col5:
        st.metric(
            label="Avg Session Length",
            value=f"{metrics['avg_session_length']} msgs"
        )
    
    with col6:
        st.metric(
            label="Today's Activity",
            value=metrics['interactions_today']
        )
    
    # Feedback breakdown
    st.markdown("#### Feedback Breakdown")
    if metrics['total_feedback'] > 0:
        st.progress(metrics['satisfaction_score'] / 100)
        st.caption(f"👍 {metrics['positive_feedback']} | 👎 {metrics['negative_feedback']}")
    else:
        st.caption("No feedback collected yet")
    
    # Mode distribution
    st.markdown("#### Mode Usage")
    total = sum(metrics['mode_distribution'].values())
    if total > 0:
        for mode, count in metrics['mode_distribution'].items():
            pct = count / total * 100
            st.caption(f"{mode.upper()}: {count} ({pct:.1f}%)")
    else:
        st.caption("No interactions yet")

def render_mini_metrics():
    """Compact metrics for sidebar"""
    metrics = calculate_metrics()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Quick Stats")
    st.sidebar.caption(f"Satisfaction: {metrics['satisfaction_score']}%")
    st.sidebar.caption(f"Sessions: {metrics['total_sessions']}")
    st.sidebar.caption(f"Interactions: {metrics['total_interactions']}")
    st.sidebar.caption(f"Avg TTFT: {metrics['avg_ttft_ms']}ms")