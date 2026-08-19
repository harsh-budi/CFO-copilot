import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)
months = pd.date_range('2022-01-01', periods=36, freq='MS')

departments = ['Sales', 'Marketing', 'Engineering', 'G&A', 'Operations']
base_revenue = 4_500_000  # $4.5M starting monthly revenue

records = []

for i, dt in enumerate(months):
    # Revenue grows ~9% annually with seasonal pattern
    trend     = 1 + 0.0075 * i
    season    = 1 + 0.07 * np.sin(2 * np.pi * (dt.month - 3) / 12)
    noise     = np.random.uniform(0.97, 1.03)
    revenue   = round(base_revenue * trend * season * noise, 0)

    # Budget is set slightly optimistically at year start
    budget_revenue = round(base_revenue * trend * 1.04, 0)

    # Prior year: same month last year (only available after month 12)
    prior_year = round(base_revenue * (1 + 0.0075*(i-12)) * season * noise, 0) if i >= 12 else None

    records.append({
        'month':          dt.strftime('%Y-%m'),
        'year':           dt.year,
        'quarter':        f"Q{(dt.month-1)//3+1}",
        'line_item':      'Revenue',
        'department':     'Total',
        'actual':         revenue,
        'budget':         budget_revenue,
        'prior_year':     prior_year
    })

    # Department expenses as % of revenue
    dept_pcts = {
        'Sales':       0.19,
        'Marketing':   0.11,
        'Engineering': 0.24,
        'G&A':         0.08,
        'Operations':  0.14
    }

    for dept, pct in dept_pcts.items():
        # Q3 2024: Engineering hires surge (this creates the margin story)
        if dept == 'Engineering' and dt.year == 2024 and dt.month in [7,8,9]:
            pct = pct * 1.18  # 18% above normal — this is the "margin compression"

        actual_exp  = round(revenue * pct * np.random.uniform(0.95, 1.06), 0)
        budget_exp  = round(revenue * pct * 1.02, 0)

        records.append({
            'month':      dt.strftime('%Y-%m'),
            'year':       dt.year,
            'quarter':    f"Q{(dt.month-1)//3+1}",
            'line_item':  'OpEx',
            'department': dept,
            'actual':     actual_exp,
            'budget':     budget_exp,
            'prior_year': None
        })

df = pd.DataFrame(records)
df.to_csv('data/financials/pl_actuals.csv', index=False)
print(f"Generated {len(df)} rows of P&L data")
print(df[df['line_item']=='Revenue'][['month','actual','budget']].tail(6).to_string(index=False))