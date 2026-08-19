from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def route_question(question: str) -> str:
    """Returns 'sql' or 'rag' for a given question"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Classify this finance question.

Return ONLY the word 'sql' or 'rag' — nothing else.

'sql' — needs exact numbers, totals, sums, comparisons from data:
  - "What was revenue in Q2?"
  - "Which department spent most?"
  - "Show me actual vs budget"
  - "What was the variance?"
  - "How much did Engineering spend?"

'rag' — needs narrative, explanation, context, reasons, analysis:
  - "Why did margins compress?"
  - "What are the key risks?"
  - "What is the hiring plan?"
  - "What drove the outperformance?"
  - "What is the revenue outlook?"
  - "Explain the EBITDA trend"

When in doubt: if the answer is a specific number → sql.
If the answer is a sentence of explanation → rag."""},
            {"role": "user", "content": question}
        ],
        temperature=0,
        max_tokens=5  # only need 'sql' or 'rag'
    )
    route = response.choices[0].message.content.strip().lower()
    return 'sql' if 'sql' in route else 'rag'

# ── Test the router ───────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("What was Q3 2024 revenue?",           "sql"),
        ("Why did margins compress in Q3?",      "rag"),
        ("Which dept spent most in 2024?",       "sql"),
        ("What is the Q4 outlook?",              "rag"),
        ("Show actual vs budget for 2024",       "sql"),
        ("What is the Engineering hiring plan?", "rag"),
    ]

    correct = 0
    for question, expected in test_cases:
        got = route_question(question)
        match = "✓" if got == expected else "✗"
        print(f"{match} '{question[:45]}' → {got} (expected {expected})")
        if got == expected:
            correct += 1

    print(f"\nRouting accuracy: {correct}/{len(test_cases)}")