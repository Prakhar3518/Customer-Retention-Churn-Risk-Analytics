import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# ---------- Load data & artifacts ----------


@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_data.csv")
    return df


@st.cache_resource
def load_artifacts():
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pickle", "rb") as f:
        scaler = pickle.load(f)
    with open("encoders.pickle", "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, encoders


df = load_data()

try:
    model, scaler, encoders = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run Customer_Churn_Analysis.ipynb first "
        "to generate model.pickle, scaler.pickle and encoders.pickle."
    )
    st.stop()

# ---------- Recreate the same preprocessing used in training ----------

data = df.copy()
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

for col, encoder in encoders.items():
    data[col] = encoder.transform(data[col].astype(str))

features = data.drop(columns=["CustomerID", "Churn"])
features_scaled = scaler.transform(features)

churn_probability = model.predict_proba(features_scaled)[:, 1]
df["Churn Probability"] = churn_probability


def risk_level(p):
    if p >= 0.66:
        return "High"
    elif p >= 0.33:
        return "Medium"
    else:
        return "Low"


df["Risk Level"] = df["Churn Probability"].apply(risk_level)

# ---------- Sidebar filters ----------

st.sidebar.header("Filters")
gender_filter = st.sidebar.multiselect(
    "Gender", options=df["Gender"].unique(), default=list(df["Gender"].unique())
)
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["ContractType"].unique(),
    default=list(df["ContractType"].unique()),
)
internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=df["InternetService"].unique(),
    default=list(df["InternetService"].unique()),
)

filtered = df[
    df["Gender"].isin(gender_filter)
    & df["ContractType"].isin(contract_filter)
    & df["InternetService"].isin(internet_filter)
]

# ---------- Header ----------

st.title("Customer Retention & Churn Risk Dashboard")
st.caption("Churn patterns, high-risk customers, revenue at risk, and retention insights")

# ---------- KPIs ----------

total_customers = len(filtered)
churned_customers = (filtered["Churn"] == "Yes").sum()
churn_rate = churned_customers / total_customers if total_customers else 0
revenue_at_risk = filtered.loc[filtered["Risk Level"] == "High", "MonthlyCharges"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{churn_rate:.1%}")
col4.metric("Revenue at Risk (Monthly)", f"${revenue_at_risk:,.2f}")

st.divider()

# ---------- Charts ----------

chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    filtered["Churn"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

with chart_col2:
    st.subheader("Churn Rate by Contract Type")
    rate_by_contract = (
        filtered.assign(is_churn=(filtered["Churn"] == "Yes").astype(int))
        .groupby("ContractType")["is_churn"]
        .mean()
    )
    st.bar_chart(rate_by_contract)

with chart_col3:
    st.subheader("Risk Level Breakdown")
    st.bar_chart(filtered["Risk Level"].value_counts())

st.divider()

# ---------- High-risk customers ----------

st.subheader("High-Risk Customers")
high_risk = (
    filtered[filtered["Risk Level"] == "High"]
    .sort_values("Churn Probability", ascending=False)
    [["CustomerID", "Age", "Tenure", "MonthlyCharges", "ContractType",
      "InternetService", "TechSupport", "Churn Probability"]]
)
st.dataframe(high_risk, use_container_width=True)

st.divider()

# ---------- Retention insights ----------

st.subheader("Retention Insights")

if total_customers:
    worst_contract = rate_by_contract.idxmax()
    worst_contract_rate = rate_by_contract.max()

    rate_by_internet = (
        filtered.assign(is_churn=(filtered["Churn"] == "Yes").astype(int))
        .groupby("InternetService")["is_churn"]
        .mean()
    )
    worst_internet = rate_by_internet.idxmax()
    worst_internet_rate = rate_by_internet.max()

    no_support_rate = (
        filtered[filtered["TechSupport"] == "No"]["Churn"].eq("Yes").mean()
        if (filtered["TechSupport"] == "No").any()
        else 0
    )

    st.markdown(
        f"""
- **{worst_contract}** contracts have the highest churn rate at **{worst_contract_rate:.1%}** — prioritize converting these customers to longer-term contracts.
- **{worst_internet}** internet service customers churn the most at **{worst_internet_rate:.1%}**.
- Customers **without Tech Support** churn at **{no_support_rate:.1%}** — offering tech support proactively could reduce churn.
- **{len(high_risk):,}** customers are currently flagged High Risk, representing **${revenue_at_risk:,.2f}** in monthly revenue at risk.
"""
    )
else:
    st.info("No customers match the selected filters.")
