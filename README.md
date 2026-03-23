# 📊 Mess Intelligence Platform

> **Startup-grade DWBI Analytics Suite for Institutional Food Systems**  

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.20+-FF6B35?logo=gradio)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Features

| Module | Description |
|--------|-------------|
| 📈 **Overview** | KPI cards, daily trend with 7-day MA, mess comparison, top-15 vendors |
| 🗑 **Wastage Analysis** | Calendar heatmap + monthly summary (proxy wastage model: `W(d) = E(d)/Ē × 100`) |
| 🔍 **Anomaly Detection** | Z-score statistical outlier flagging with configurable σ threshold |
| 📐 **Benford's Law** | Forensic first-digit χ² goodness-of-fit analysis |
| 🕸 **Network Analysis** | Weighted bipartite mess–vendor graph with Louvain community detection |
| 🔗 **Association Rules** | FP-Growth frequent pattern mining (configurable support/confidence) |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/MaitreyaGanu/Mess-Case-Study
cd Mess-Case-Study

# 2. Install
pip install -r Code/requirements.txt

# 3. Run
python Code/app.py
# → Open http://localhost:7860
```

---

## 📂 CSV Format

Your file needs these columns (names are flexible — auto-detected):

| Column | Required | Example |
|--------|----------|---------|
| `date` | ✅ | `2025-08-01` |
| `amount` | ✅ | `150000` |
| `vendor` | Recommended | `Amma Vegetables` |
| `mess` / `unit` | Recommended | `CDH-1` |


---

## 🔬 Methodology

- **Wastage Proxy**: `W(d) = (E(d) / Ē_month) × k` where k = 100 kg baseline constant
- **Benford Test**: χ² goodness-of-fit, df=8, α=0.05 (χ²_critical = 15.51)
- **Network**: Louvain modularity maximization on weighted bipartite graph
- **ARM**: FP-Growth with min_support=0.05, min_confidence=0.60, min_lift=1.0

---

## 📖 Paper Reference

*"Optimizing Institutional Food Systems via Transactional Data: A Residential University Case Study"*  
Maitreya Sameer Ganu (IMS23099) · IISER Thiruvananthapuram · Dec 2025  
Supervisor: Dr. Zakaria Laskar, School of Data Science

---

## 📜 License
MIT License — free to use, modify, deploy.

---
# Reducing Food Wastage and Procurement Loss in a Student Mess  
## A Data Warehousing and Business Intelligence Case Study

**Author:** Maitreya Sameer Ganu (IMS23099)  
**Supervisor:** Dr. Zakaria Laskar  
**Institution:** Indian Institute of Science Education and Research (IISER), Thiruvananthapuram  
**Course Project:** January–April 2026  

---

## Overview

This repository contains the datasets, SQL workflows, and Python-based analytical code used in the academic case study:

**“Reducing Food Wastage and Procurement Loss in a Student Mess Using Data Warehousing and Business Intelligence Techniques.”**

The project applies **data warehousing (DW)** and **business intelligence (BI)** methodologies to real-world institutional mess expenditure data to identify inefficiencies, wastage-prone periods, and procurement risk patterns in a **non-profit, student-run food system**.

Due to the absence of direct food wastage measurements, the analysis adopts a **proxy-based estimation framework** grounded in expenditure behavior. The objective is not profit maximization, but **operational efficiency and waste reduction**.

---

## Data Source

- **Source:**Provided in he description 
- **Portal:** http://mess.iisertvm.ac.in  
- **Data Type:** Vendor-level expenditure records  
- **Time Period:** August–December 2025

Each record corresponds to a **vendor payment transaction**, not individual food consumption events. Multiple vendor bills may be settled on the same payment date.

---

## Repository Structure

```
.
├── Code/
│   ├── data_preprocessing.py
│   ├── september_analysis.py
│   ├── october_analysis.py
│   ├── november_analysis.py
│   ├── december_analysis.py
│   ├── august_analysis.py
│   ├── benford_analysis.py
|   |── updatedAssociationRules.py
|   |──app.py
|   |──requirements.txt  
│   └── network_analysis.py
|    
│
├── README.md
```



---

## Code Description

All analysis is implemented in **Python** and organized modularly by analytical task and temporal scope.

### `data_preprocessing.py`
- Cleans raw CSV data extracted from the mess portal  
- Handles missing or inconsistent dates  
- Normalizes vendor names  
- Aggregates vendor-level transactions into daily expenditure values  

This script forms the **ETL preprocessing stage** of the DWBI pipeline.

---

### Monthly Analysis Scripts  
(`august_analysis.py`, `september_analysis.py`, `october_analysis.py`, `november_analysis.py`, `december_analysis.py`)

- Perform **month-wise exploratory data analysis**
- Compute daily total expenditure
- Estimate **relative food wastage risk** using proxy-based scaling
- Identify expenditure spikes and temporal patterns
- Correlate observed trends with academic calendar events

These scripts support **time-series analysis** and comparative month-to-month evaluation.

---

### `benford_analysis.py`

- Applies **Benford’s Law** to vendor expenditure amounts
- Analyzes first-digit distributions
- Flags statistically unusual patterns for further administrative review  

This analysis is used as a **business intelligence anomaly detection heuristic**, not as a definitive fraud detection mechanism.

---

### `network_analysis.py`

- Constructs a **vendor–mess unit interaction network**
- Examines structural dependencies in procurement
- Identifies high-dependency vendors and central nodes

This component supports **procurement risk assessment** and supplier dependency analysis.

---

## Analytical Methodology

The project follows a complete **data warehousing and knowledge discovery pipeline**:

1. Data Cleaning and Validation  
2. Temporal and Entity-Level Data Integration  
3. Aggregation and Feature Transformation  
4. Temporal Trend Analysis  
5. Statistical Distribution Analysis  
6. Network-Based Structural Analysis  
7. Knowledge Presentation via Visualizations  

All interpretations are made with explicit acknowledgment that **payment dates do not correspond to actual consumption dates**.

---

## Wastage Estimation Model

Due to the absence of direct wastage data:

- A baseline wastage of **100 kg/day** is assumed
- Estimated wastage is scaled proportionally to:

Estimated values represent **relative wastage risk indicators**, not absolute food waste quantities.
---

## Limitations

- Payment date ≠ consumption date  
- Daily expenditure spikes may reflect bulk settlements  
- No item-level or quantity-wise procurement data  
- Proxy-based wastage estimation relies on proportional assumptions  

These constraints are consistently acknowledged throughout the analysis.

---

## Report

A detailed academic report describing the data warehouse design, methodology, analysis, and limitations accompanies this repository.

**Title:**  
*Reducing Food Wastage and Procurement Loss in a Student Mess Using Data Warehousing and Business Intelligence Techniques*

---

## Intended Use

This repository is intended for:
- Academic evaluation  
- Reproducible research  
- Educational use in DWBI and applied data mining  


