# Veridi Logistics — Interactive Dashboard

Live dashboard for the Veridi Logistics Delivery Performance Audit.

## Dashboard Link

[https://veridi-dashboard-d9hvcucgczhm9hk6pwqja7.streamlit.app/](https://veridi-dashboard-d9hvcucgczhm9hk6pwqja7.streamlit.app/)

## Pages

| Page | Content |
|---|---|
| Overview | KPI cards, delivery performance donut, delay distribution histogram, key findings |
| Geographic Analysis | Brazil choropleth map, state rankings bar chart, state detail table |
| Sentiment Analysis | Review score by delivery class, delay vs review score scatter plot |
| Category Breakdown | Top 20 worst-performing product categories with interactive slider |
| Trend Analysis | Monthly volume, late rate and satisfaction trend, quarterly heatmap |
| ML Prediction Model | Feature importance chart, model metrics, business recommendations |

## Dataset

Olist Brazilian E-Commerce Public Dataset — [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Running Locally

```bash
pip install streamlit pandas numpy plotly scikit-learn
streamlit run app.py
```

Place all Olist CSV files in the same directory as `app.py`.
