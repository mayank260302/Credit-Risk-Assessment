# Credit Risk Assessment Engine

This project implements an end-to-end credit risk assessment pipeline using
industry-standard banking methodologies.

The system covers Probability of Default (PD) modeling, feature engineering
using Weight of Evidence (WoE) and Information Value (IV), credit scorecard
generation, and Expected Loss calculation.

---

## Features

- Probability of Default (PD) model using Logistic Regression
- WoE and IV based feature selection
- Credit scorecard generation
- Expected Loss calculation (PD × LGD × EAD)
- Outputs designed for Power BI dashboards

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Power BI
- Git & GitHub

---

## Project Structure

Credit-Risk-Assessment/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│ ├── data_preprocessing.py
│ ├── pd_model.py
│ ├── woe_iv.py
│ ├── scorecard.py
│ └── expected_loss.py
├── outputs/
└── dashboard/  


---

## Dataset

Home Credit Default Risk Dataset  
Source: https://www.kaggle.com/c/home-credit-default-risk

The dataset is not included in this repository due to licensing restrictions.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
