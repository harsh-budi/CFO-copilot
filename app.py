import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Auto-build vector store on startup (critical for Streamlit Cloud)
from startup import ensure_vector_store
ensure_vector_store()

import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3

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
            if len(chart_data) > 1:
                df_chart = chart_data.copy()
                x_col = df_chart.columns[0]  # first column is always the category

            # Check if both actual and budget columns exist
            # → grouped bar chart showing actual vs budget side by side
                if 'actual' in df_chart.columns and 'budget' in df_chart.columns:
                    fig = px.bar(
                        df_chart.melt(
                            id_vars=[x_col],
                            value_vars=['actual', 'budget'],
                            var_name='Type',
                            value_name='Amount'
                        ),
                        x=x_col,
                        y='Amount',
                        color='Type',
                        barmode='group',
                        title=user_input[:60],
                        color_discrete_map={
                            'actual': '#185FA5',
                            'budget': '#6BAED6'
                        },
                        labels={'Amount': 'Revenue ($)', x_col: x_col.title()}
                    )
                        # Add variance % as text annotation if column exists
                    if 'variance_pct' in df_chart.columns:
                        for i, row in df_chart.iterrows():
                             color = '#27500A' if row['variance_pct'] >= 0 else '#791F1F'
                            fig.add_annotation(
                                x=row[x_col],
                                y=max(row['actual'], row['budget']) * 1.03,
                                text=f"{row['variance_pct']:+.1f}%",
                                showarrow=False,
                                font=dict(size=11, color=color),
                                xref='x', yref='y'
                            )
                    fig.update_layout(legend_title_text='')
                    st.plotly_chart(fig, use_container_width=True)

            # Check if variance_pct column exists alone
            # → single bar chart colored by positive/negative variance
                elif 'variance_pct' in df_chart.columns:
                    df_chart['color'] = df_chart['variance_pct'].apply(
                        lambda v: 'Above Budget' if v >= 0 else 'Below Budget'
                    )
                    fig = px.bar(
                        df_chart,
                        x=x_col,
                        y='variance_pct',
                        color='color',
                        title=user_input[:60],
                        color_discrete_map={
                        'Above Budget': '#1D9E75',
                        'Below Budget': '#D85A30'
                        },
                        labels={'variance_pct': 'Variance (%)', x_col: x_col.title()}
                    )
                    fig.add_hline(y=0, line_dash='dash', line_color='gray')
                    st.plotly_chart(fig, use_container_width=True)

            # Default fallback → simple bar chart with first numeric column
                else:
                    numeric_cols = df_chart.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        fig = px.bar(
                            df_chart,
                            x=x_col,
                            y=numeric_cols[0],
                            title=user_input[:60],
                            color_discrete_sequence=['#185FA5']
                        )
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            pass

    # Save assistant message to history
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   answer,
        "source":    source,
        "route":     route,
        "chart_data": chart_data
    })