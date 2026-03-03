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
│   └── network_analysis.py
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


