import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3, sys, os
sys.path.insert(0, 'scripts')

from rag_pipeline import build_rag_chain
from text_to_sql import run_sql_query
from router import route_question

st.set_page_config(
    page_title="CFO Co-Pilot",
    page_icon="💼",
    layout="wide"
)

# ── Load data for the KPI sidebar ─────────────────────────────
@st.cache_data
def load_kpis():
    try:
        df = pd.read_csv('data/financials/pl_actuals.csv')
        rev = df[(df['line_item']=='Revenue') & (df['year']==2024)]['actual']
        bud = df[(df['line_item']=='Revenue') & (df['year']==2024)]['budget']
        opex_depts = df[(df['line_item']=='OpEx') & (df['year']==2024)]
        total_opex = opex_depts['actual'].sum()
        gross_profit = rev.sum() - total_opex * 0.38
        margin = (rev.sum() - total_opex) / rev.sum() * 100
        return {
            'revenue': rev.sum(),
            'budget':  bud.sum(),
            'margin':  round(margin, 1),
            'var_pct': round((rev.sum() - bud.sum()) / bud.sum() * 100, 1)
        }
    except:
        return None

# ── Build RAG chain once and cache it ─────────────────────────
# This prevents rebuilding the chain on every user message
@st.cache_resource
def get_rag_chain():
    return build_rag_chain()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### CFO Co-Pilot")
    st.caption("AI-powered financial Q&A")
    st.divider()

    kpis = load_kpis()
    if kpis:
        st.markdown("**2024 Portfolio KPIs**")
        st.metric(
            "Revenue YTD",
            f"${kpis['revenue']/1e6:.1f}M",
            delta=f"{kpis['var_pct']:+.1f}% vs budget"
        )
        st.metric("EBITDA Margin", f"{kpis['margin']:.1f}%")

    st.divider()
    st.markdown("**Try asking:**")

    example_questions = [
        "What drove Q3 2024 margin compression?",
        "What was total revenue in Q3 2024?",
        "Which department exceeded budget most in 2024?",
        "What is the Q4 revenue outlook?",
        "What was FY2023 EBITDA margin?",
        "Show revenue vs budget variance by quarter in 2024",
        "What is the Engineering hiring plan?",
        "What are the key risks for Q4?"
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_q = q

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop('pending_q', None)

# ── Main chat area ─────────────────────────────────────────────
st.title("CFO Co-Pilot")
st.caption("Ask questions about 3 years of P&L, budget, and actuals data")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("source"):
            st.caption(f"Source: {msg['source']} · Route: {msg.get('route','').upper()}")
        if msg.get("chart_data") is not None:
            try:
                df_chart = msg["chart_data"]
                if len(df_chart) > 1 and len(df_chart.columns) >= 2:
                    fig = px.bar(df_chart, x=df_chart.columns[0],
                                 y=df_chart.columns[1],
                                 color_discrete_sequence=['#185FA5'])
                    st.plotly_chart(fig, use_container_width=True)
            except:
                pass

# Get user input — either typed or clicked from sidebar
user_input = st.chat_input("Ask a financial question...")
if "pending_q" in st.session_state:
    user_input = st.session_state.pop("pending_q")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial data..."):
            route = route_question(user_input)

            if route == 'sql':
                result = run_sql_query(user_input)
                if result["success"]:
                    df_result = result["data"]
                    answer = f"**Query result:**\n\n{result['result_text']}"
                    chart_data = df_result if len(df_result) > 1 else None
                    source = "Structured financial data (SQL query)"
                else:
                    answer = f"Could not retrieve that data. Error: {result['error']}"
                    chart_data = None
                    source = "SQL (error)"
            else:
                rag_chain = get_rag_chain()
                answer = rag_chain.invoke(user_input)
                chart_data = None
                source = "Financial documents & commentary (RAG)"

        st.write(answer)
        st.caption(f"Source: {source} · Route: {route.upper()}")

        # Show chart for SQL results with multiple rows
        if chart_data is not None:
            try:
                if len(chart_data) > 1 and len(chart_data.columns) >= 2:
                    fig = px.bar(chart_data, x=chart_data.columns[0],
                                 y=chart_data.columns[1],
                                 title=user_input[:60],
                                 color_discrete_sequence=['#185FA5'])
                    st.plotly_chart(fig, use_container_width=True)
            except:
                pass

    # Save assistant message to history
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   answer,
        "source":    source,
        "route":     route,
        "chart_data": chart_data
    })