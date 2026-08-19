import sqlite3
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ── Load P&L data into an in-memory SQLite database ──────────
# In-memory means the database exists only while the script runs.
# It is re-created every time from the CSV — always fresh data.
def get_db_connection():
    conn = sqlite3.connect(':memory:')
    df = pd.read_csv('data/financials/pl_actuals.csv')
    df.to_sql('financials', conn, index=False, if_exists='replace')
    return conn

# ── Database schema sent to GPT-4 ────────────────────────────
# This is critical — GPT-4 needs to know exactly what columns exist
# and what values they contain to write correct SQL.
# The more detail you give here, the better the SQL will be.
DB_SCHEMA = """
SQLite table: financials

Columns:
- month        TEXT    format 'YYYY-MM' e.g. '2024-07'
- year         INTEGER values: 2022, 2023, 2024
- quarter      TEXT    values: 'Q1', 'Q2', 'Q3', 'Q4'
- line_item    TEXT    values: 'Revenue' or 'OpEx'
- department   TEXT    values: 'Total','Sales','Marketing',
                       'Engineering','G&A','Operations'
                       (use 'Total' for company-wide Revenue)
- actual       REAL    dollar amount — the real figure
- budget       REAL    dollar amount — the planned figure
- prior_year   REAL    dollar amount — same period last year (nullable)

Examples:
- Revenue for a month: SELECT actual FROM financials
  WHERE line_item='Revenue' AND month='2024-07'
- Total Q3 2024 revenue: SELECT SUM(actual) FROM financials
  WHERE line_item='Revenue' AND year=2024 AND quarter='Q3'
- Engineering Q3 spend: SELECT SUM(actual) FROM financials
  WHERE line_item='OpEx' AND department='Engineering'
  AND year=2024 AND quarter='Q3'
- Budget vs actual variance: SELECT
  SUM(actual) as actual, SUM(budget) as budget,
  ROUND((SUM(actual)-SUM(budget))/SUM(budget)*100,1) as variance_pct
  FROM financials WHERE line_item='Revenue' AND year=2024
"""

def generate_sql(question: str) -> str:
    """Ask GPT-4 to convert a natural language question to SQL"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a SQL expert.
Convert the financial question to a SQLite SQL query.

Schema:
{DB_SCHEMA}

Rules:
- Return ONLY the SQL query — no explanation, no markdown, no backticks
- Always use SUM() for revenue and expense totals
- Use ROUND(..., 0) for dollar amounts, ROUND(...,1) for percentages
- Filter line_item appropriately: Revenue rows for revenue questions,
  OpEx rows for expense questions"""},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def run_sql_query(question: str) -> dict:
    """Generate SQL from question and execute it, return result"""
    sql = generate_sql(question)
    print(f"Generated SQL: {sql}")

    try:
        conn = get_db_connection()
        result_df = pd.read_sql(sql, conn)
        conn.close()

        if result_df.empty:
            return {"success": False, "error": "Query returned no results", "sql": sql}

        return {
            "success": True,
            "sql": sql,
            "data": result_df,
            "result_text": result_df.to_string(index=False)
        }
    except Exception as e:
        return {"success": False, "error": str(e), "sql": sql}

# ── Test the text-to-SQL pipeline ─────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "What was total revenue in Q3 2024?",
        "What was Engineering expense in Q3 2024?",
        "Show revenue actual vs budget variance by quarter in 2024",
        "Which department had the highest actual spend in 2024?",
        "What was the revenue growth from 2022 to 2023?"
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = run_sql_query(q)
        if result["success"]:
            print(f"Result:\n{result['result_text']}")
        else:
            print(f"Error: {result['error']}")