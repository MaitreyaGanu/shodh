# =====================================================
# COMPLETE PIPELINE: APRIORI + FP-GROWTH + ASSOCIATION RULES
# =====================================================

import os
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
from urllib.parse import quote_plus

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# -----------------------------------------------------
# 1. DATABASE CONNECTION
# Use environment variable — never hardcode credentials
# Set with: export DB_PASSWORD="your_password"
# -----------------------------------------------------

password = quote_plus(os.environ.get("DB_PASSWORD", ""))

engine = create_engine(
    f"mysql+pymysql://root:{password}@localhost/mess_dw"
)


# -----------------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------------

query = """
SELECT
    d.full_date,
    mu.mess_unit_name,
    v.vendor_name
FROM fact_expense f
JOIN dim_date d ON f.date_id = d.date_id
JOIN dim_vendor v ON f.vendor_id = v.vendor_id
JOIN dim_mess_unit mu ON f.mess_unit_id = mu.mess_unit_id
"""

df = pd.read_sql(query, engine)


# -----------------------------------------------------
# 3. NORMALIZATION
#
# Key fixes vs original:
#   - cdh pattern now strips bare "cdh" too (digit made optional)
#   - cafe pattern unchanged (already had optional digit)
#   - manual alias table merges known split variants
#     e.g. "Milan Agencies" / "Milan Agencies Milk" -> one item
# -----------------------------------------------------

# Manual alias table: maps normalized name -> canonical name
# Add more entries here as you discover splits in your data
VENDOR_ALIASES = {
    "Milan Agencies":        "Milan Agencies Milk",
    "C H M Traders Nandini": "C H M Traders Nandini Milk",
    "Amma Vegitables":       "Amma Vegetables",   # extra safety
    "Priya Vegitables Chalai": "Priya Vegetables Chalai",
}

def normalize_vendor(name):
    name = name.lower().strip()
    # FIX: \d* (zero or more digits) instead of \d+ (one or more)
    # This catches "CDH", "CDH-1", "CDH_2", "CDH 1" etc.
    name = re.sub(r'cdh[-_\s]*\d*', '', name)
    name = re.sub(r'cafe[-_\s]*\d*', '', name)
    name = re.sub(r'[-_/()]', ' ', name)
    name = name.replace("vegitables", "vegetables")
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.title()
    # Apply manual aliases
    return VENDOR_ALIASES.get(name, name)

df["vendor_name"] = df["vendor_name"].apply(normalize_vendor)


# -----------------------------------------------------
# 4. TRANSACTIONS
#
# Each transaction = all unique vendors for a (date, mess_unit) pair.
# FIX: use set() inside the groupby to deduplicate vendors
# that map to the same name after normalization.
# -----------------------------------------------------

transactions = (
    df.groupby(["full_date", "mess_unit_name"])["vendor_name"]
    .apply(lambda x: list(set(x)))   # set() removes intra-transaction duplicates
    .tolist()
)

# Sanity check
print(f"Total transactions : {len(transactions)}")
print(f"Unique vendors     : {len(set(v for t in transactions for v in t))}")


# -----------------------------------------------------
# 5. ONE-HOT ENCODING
# -----------------------------------------------------

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_array, columns=te.columns_)


# -----------------------------------------------------
# 6. FREQUENT ITEMSETS — APRIORI + FP-GROWTH
# -----------------------------------------------------

apriori_sets = apriori(basket, min_support=0.05, use_colnames=True)
fp_sets      = fpgrowth(basket, min_support=0.05, use_colnames=True)

# Verify both algorithms agree
closed_ap = set(
    frozenset(r['itemsets'])
    for _, r in apriori_sets.iterrows()
    if not any(
        r['itemsets'] < r2['itemsets'] and r['support'] == r2['support']
        for _, r2 in apriori_sets.iterrows()
    )
)
closed_fp = set(
    frozenset(r['itemsets'])
    for _, r in fp_sets.iterrows()
    if not any(
        r['itemsets'] < r2['itemsets'] and r['support'] == r2['support']
        for _, r2 in fp_sets.iterrows()
    )
)

max_ap = set(
    frozenset(r['itemsets'])
    for _, r in apriori_sets.iterrows()
    if not any(r['itemsets'] < r2['itemsets'] for _, r2 in apriori_sets.iterrows())
)
max_fp = set(
    frozenset(r['itemsets'])
    for _, r in fp_sets.iterrows()
    if not any(r['itemsets'] < r2['itemsets'] for _, r2 in fp_sets.iterrows())
)

print(f"\nClosed itemsets match  : {closed_ap == closed_fp}")
print(f"Maximal itemsets match : {max_ap == max_fp}")
assert closed_ap == closed_fp, "Algorithm mismatch — check normalization"
assert max_ap == max_fp,       "Algorithm mismatch — check normalization"


# -----------------------------------------------------
# 7. ASSOCIATION RULES  (single, clean generation)
# -----------------------------------------------------

rules = association_rules(
    apriori_sets,
    metric="confidence",
    min_threshold=0.6
)

# Strong rules only: lift > 1
rules = rules[rules["lift"] > 1].copy()
rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

print(f"\nStrong rules generated : {len(rules)}")


# -----------------------------------------------------
# 8. PRINT ALL RULES — no truncation
# -----------------------------------------------------

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n" + "="*60)
print("ALL STRONG ASSOCIATION RULES")
print("="*60)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

print("\n" + "="*60)
print("DETAILED RULE INTERPRETATION")
print("="*60)
for i, row in enumerate(rules.itertuples(), 1):
    print(f"\nRule {i}:")
    print(f"  {set(row.antecedents)} => {set(row.consequents)}")
    print(f"  Support    : {round(row.support, 4)}")
    print(f"  Confidence : {round(row.confidence, 4)}")
    print(f"  Lift       : {round(row.lift, 4)}")

rules.to_csv("strong_rules_full.csv", index=False)
print("\n✔ Saved: strong_rules_full.csv")


# -----------------------------------------------------
# 9. VISUALIZATIONS
# -----------------------------------------------------

# --- Top rules bar chart ---
top_rules = rules.head(10)
labels = [
    f"{list(a)} → {list(c)}"
    for a, c in zip(top_rules["antecedents"], top_rules["consequents"])
]
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_rules)), top_rules["lift"])
plt.yticks(range(len(top_rules)), labels, fontsize=8)
plt.xlabel("Lift")
plt.title("Top 10 Strong Association Rules")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# --- Association rule network ---
G_rules = nx.DiGraph()
for _, row in top_rules.iterrows():
    for a in row["antecedents"]:
        for c in row["consequents"]:
            G_rules.add_edge(a, c, weight=round(row["lift"], 2))

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_rules, k=0.8, seed=42)  # seed for reproducibility
nx.draw(G_rules, pos, with_labels=True, node_size=3000, font_size=8)
nx.draw_networkx_edge_labels(
    G_rules, pos,
    edge_labels=nx.get_edge_attributes(G_rules, 'weight')
)
plt.title("Association Rule Network (Top 10 Rules)")
plt.tight_layout()
plt.show()


# --- Item frequency bar chart ---
item_freq = basket.sum().sort_values(ascending=False)
top_items = item_freq.head(10)

plt.figure(figsize=(10, 5))
plt.bar(top_items.index, top_items.values)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Frequency (number of transactions)")
plt.title("Top 10 Most Frequent Vendors")
plt.tight_layout()
plt.show()


# --- Co-occurrence heatmap ---
top_items_list = top_items.index.tolist()
co_matrix = pd.DataFrame(0, index=top_items_list, columns=top_items_list)

for transaction in transactions:
    for i in top_items_list:
        if i in transaction:
            for j in top_items_list:
                if j in transaction:
                    co_matrix.loc[i, j] += 1

plt.figure(figsize=(10, 8))
sns.heatmap(co_matrix, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Item Co-occurrence Heatmap")
plt.tight_layout()
plt.show()
