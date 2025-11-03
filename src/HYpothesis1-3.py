# nfhs_imr_analysis.py
# Reproducible analysis pipeline for NFHS IMR vs Wealth/Residence/Maternal Age
# Author: ChatGPT Assistant (for user)
# Run with: python nfhs_imr_analysis.py
# Requirements: pandas, numpy, matplotlib, statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# -------------------------
# 0) File paths & outputs
# -------------------------
HP_CSV = "nfhs5_subset_hp.csv"
MZ_CSV = "nfhs5_subset_mz.csv"

OUT_CHILD_CSV = "nfhs_child_level_final.csv"
OUT_WEALTH_PLOT = "wealth_vs_imr_plot.png"
OUT_RESIDENCE_PLOT = "residence_imr_plot.png"
OUT_MOTHERAGE_PLOT = "mother_age_imr_plot.png"
OUT_REG_TABLES = "regression_tables_by_state.csv"

# -------------------------
# 1) Load datasets
# -------------------------
hp = pd.read_csv(HP_CSV)
mz = pd.read_csv(MZ_CSV)

hp['StateName'] = 'Himachal Pradesh'
mz['StateName'] = 'Mizoram'

# Combine (we'll process each state separately when needed)
df = pd.concat([hp, mz], ignore_index=True)

# -------------------------
# 2) Identify child columns
# -------------------------
# NFHS splits child's info into repeated columns: 'child is alive', 'current age of child', etc.
child_alive_cols = [c for c in df.columns if 'child is alive' in c.lower()]
child_age_cols = [c for c in df.columns if 'current age of child' in c.lower()]

# Confirm equal length (we expect same number of child slots)
assert len(child_alive_cols) == len(child_age_cols), "Mismatch child columns length"

# -------------------------
# 3) Create child-level dataset
# -------------------------
# We will create one row per child. Infant death inference rules:
# - If 'child is alive' == 1 -> infant_died = 0
# - If 'child is alive' == 0:
#     - if 'current age of child' is present and < 1 -> infant_died = 1
#     - if 'current age of child' is missing -> infant_died = 1 (we treat missing age for deceased as death)
# - If both alive & age missing -> skip (no child record)
#
# We copy a few mother-level vars to each child record: wealth index, residence, mother age

records = []
for idx, row in df.iterrows():
    for alive_col, age_col in zip(child_alive_cols, child_age_cols):
        alive = row.get(alive_col)
        age = row.get(age_col)
        # If both missing, no child in this slot
        if pd.isna(alive) and pd.isna(age):
            continue

        infant_died = None
        # If alive is present:
        if pd.notna(alive):
            if int(alive) == 1:
                infant_died = 0
            elif int(alive) == 0:
                # child not alive
                if pd.notna(age):
                    # if age recorded, mark as infant death if age < 1
                    try:
                        age_val = float(age)
                    except:
                        age_val = np.nan
                    if pd.notna(age_val) and age_val < 1:
                        infant_died = 1
                    else:
                        # died but age >= 1 => not infant death
                        infant_died = 0
                else:
                    # deceased but no age recorded -> treat as infant death (consistent with NFHS pattern)
                    infant_died = 1
        else:
            # Alive missing but age present -> treat as alive (age must be current age)
            if pd.notna(age):
                infant_died = 0

        if infant_died is None:
            continue

        records.append({
            'StateName': row['StateName'],
            'wealth_index': row.get('wealth index combined', np.nan),
            'residence_type': row.get('type of place of residence', np.nan),
            'mother_age': row.get("respondent's current age", np.nan),
            'infant_died': int(infant_died)
        })

child_df = pd.DataFrame(records)

# Drop rows with missing key covariates (we keep as much as possible; if you want strict subset, change here)
child_df = child_df.dropna(subset=['wealth_index', 'residence_type', 'mother_age', 'infant_died']).copy()
child_df['wealth_index'] = child_df['wealth_index'].astype(int)
child_df['residence_type'] = child_df['residence_type'].astype(int)
child_df['mother_age'] = pd.to_numeric(child_df['mother_age'], errors='coerce')
child_df['infant_died'] = child_df['infant_died'].astype(int)

# Save child-level dataset for reproducibility
child_df.to_csv(OUT_CHILD_CSV, index=False)
print(f"Saved child-level file: {OUT_CHILD_CSV}")

# Quick summary counts
summary = child_df.groupby('StateName')['infant_died'].agg(['count','sum'])
summary['death_rate'] = summary['sum'] / summary['count']
print("\nChild-level summary by state:")
print(summary)

# -------------------------
# 4) Modeling functions
# -------------------------
def run_logit(df_sub, predictors, weights=None, freq_weights=False):
    """
    Run logistic regression using statsmodels Logit.
    predictors: list of column names used as X (no constant)
    weights: array-like same length as df_sub (optional)
    freq_weights: if True, pass weights as freq_weights argument (statsmodels supports that)
    Returns: fitted model or None if not enough variation
    """
    X = df_sub[predictors].copy()
    X = sm.add_constant(X)
    y = df_sub['infant_died']
    if y.nunique() < 2:
        return None
    if weights is None:
        model = sm.Logit(y, X).fit(disp=False)
    else:
        # Use freq_weights in statsmodels (note: later versions change arguments; this works typically)
        model = sm.Logit(y, X).fit(freq_weights=weights, disp=False)
    return model

# -------------------------
# 5) Simple logistic per state: infant_died ~ wealth_index
# -------------------------
models_simple = {}
for state in child_df['StateName'].unique():
    sub = child_df[child_df['StateName'] == state].copy()
    m = run_logit(sub, ['wealth_index'])
    models_simple[state] = m
    if m is not None:
        print(f"\n=== Simple Logit ({state}) ===")
        print(m.summary2().tables[1])
    else:
        print(f"\nNo variation in {state} for simple model.")

# Save a plot: Predicted probability vs Wealth Index (1..5)
wealth_vals = np.arange(1,6)

plt.figure(figsize=(8,5))
for state, model in models_simple.items():
    if model is None:
        continue
    # produce predictions by setting X values at wealth_vals
    Xpred = sm.add_constant(pd.DataFrame({'wealth_index': wealth_vals}))
    preds = model.predict(Xpred)
    plt.plot(wealth_vals, preds, marker='o', label=state)

plt.xticks(wealth_vals)
plt.xlabel("Wealth Index (1=Poorest .. 5=Richest)")
plt.ylabel("Predicted Probability of Infant Death")
plt.title("Predicted Infant Death Probability by Wealth Index")
plt.legend()
plt.grid(True)
plt.savefig(OUT_WEALTH_PLOT, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved wealth plot: {OUT_WEALTH_PLOT}")

# -------------------------
# 6) Residence type model: infant_died ~ residence_type (statewise)
# -------------------------
models_res = {}
for state in child_df['StateName'].unique():
    sub = child_df[child_df['StateName'] == state].copy()
    m = run_logit(sub, ['residence_type'])
    models_res[state] = m

# Plot predicted probabilities for urban(1) and rural(2)
res_vals = [1,2]
plt.figure(figsize=(7,5))
for state, model in models_res.items():
    if model is None:
        continue
    Xpred = sm.add_constant(pd.DataFrame({'residence_type': res_vals}))
    preds = model.predict(Xpred)
    plt.plot(res_vals, preds, marker='o', label=state)
plt.xticks(res_vals, ['Urban (1)','Rural (2)'])
plt.xlabel("Residence Type")
plt.ylabel("Predicted Probability of Infant Death")
plt.title("Predicted Infant Death Probability by Residence Type")
plt.legend()
plt.grid(True)
plt.savefig(OUT_RESIDENCE_PLOT, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved residence plot: {OUT_RESIDENCE_PLOT}")

# -------------------------
# 7) Multivariate model: infant_died ~ wealth_index + residence_type + mother_age
# -------------------------
models_multi = {}
for state in child_df['StateName'].unique():
    sub = child_df[child_df['StateName'] == state].copy()
    m = run_logit(sub, ['wealth_index','residence_type','mother_age'])
    models_multi[state] = m
    if m is not None:
        print(f"\n=== Multivariate Logit ({state}) ===")
        print(m.summary2().tables[1])

# -------------------------
# 8) Rare-event adjustment (Mizoram): weighted logistic
# -------------------------
# We'll compute simple inverse-frequency weights so that events (deaths) get more influence.
mz_df = child_df[child_df['StateName'] == 'Mizoram'].copy()
n_total = len(mz_df)
n_events = mz_df['infant_died'].sum()
n_nonevents = n_total - n_events

# Compute frequency-type weights (inverse frequency approach)
# weight_event = n_total / (2 * n_events), weight_non_event = n_total / (2 * n_non_events)
if n_events > 0:
    event_weight = n_total / (2.0 * n_events)
    non_event_weight = n_total / (2.0 * n_nonevents) if n_nonevents > 0 else 1.0
    mz_df['weights'] = np.where(mz_df['infant_died'] == 1, event_weight, non_event_weight)
else:
    mz_df['weights'] = 1.0

# Weighted logistic for mother's age in Mizoram
X = sm.add_constant(mz_df['mother_age'])
y = mz_df['infant_died']
weights = mz_df['weights']
if y.nunique() > 1:
    weighted_model_age = sm.Logit(y, X).fit(freq_weights=weights, disp=False)
    print("\n=== Weighted Logistic (Mizoram): infant_died ~ mother_age ===")
    print(weighted_model_age.summary2().tables[1])
else:
    weighted_model_age = None
    print("No events or no variation for Mizoram mother_age model.")

# Plot the observed points + fitted weighted curve
if weighted_model_age is not None:
    ages = np.linspace(18, 45, 100)
    Xpred = sm.add_constant(pd.DataFrame({'mother_age': ages}))
    preds = weighted_model_age.predict(Xpred)

    plt.figure(figsize=(8,6))
    # scatter observed points with slight jitter for visualization (colored by event)
    jitter = np.random.uniform(-0.25, 0.25, size=len(mz_df))
    plt.scatter(mz_df['mother_age'] + jitter, mz_df['infant_died'],
                c=mz_df['infant_died'], cmap='gray', alpha=0.25, s=12, label='Observed (0=alive,1=death)')
    plt.plot(ages, preds, color='navy', lw=2, label='Weighted Logistic Fit')
    plt.xlabel("Mother's age (years)")
    plt.ylabel("Infant death (0/1) / Predicted probability")
    plt.title("Mizoram: Observed data & weighted logistic fit (mother age)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_MOTHERAGE_PLOT, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved mother age plot: {OUT_MOTHERAGE_PLOT}")

# -------------------------
# 9) Export regression tables to CSV for the LaTeX report
# -------------------------
rows = []
for state, model in models_multi.items():
    if model is None:
        continue
    table = model.summary2().tables[1].reset_index().rename(columns={'index':'term'})
    table['state'] = state
    rows.append(table[['state','term','Coef.','Std.Err.','z','P>|z|','[0.025','0.975]']])
if len(rows) > 0:
    all_tables = pd.concat(rows, ignore_index=True)
    # Normalize column names
    all_tables.columns = ['state','term','coef','std_err','z','pval','ci_lower','ci_upper']
    all_tables.to_csv(OUT_REG_TABLES, index=False)
    print(f"Saved regression tables: {OUT_REG_TABLES}")

print("\nAnalysis complete. Outputs saved to current directory.")
