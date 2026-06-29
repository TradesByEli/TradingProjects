import streamlit as st

st.set_page_config(page_title="Position Size Calculator", layout="wide")

RISK_PRESETS = {
    "2%": 0.02,
    "5% (B)": 0.05,
    "7.5%": 0.075,
    "10%": 0.10,
    "12.5%": 0.125,
    "15% (B+)": 0.15,
    "20%": 0.20,
    "25%": 0.25,
    "30% (A-)": 0.30,
    "40%": 0.40,
    "50% (A)": 0.50,
    "80%": 0.80,
    "90%": 0.90,
    "100% (A+)": 1.00,
}

st.title("⚡ Position Size Calculator")
st.caption("All sizing details shown directly in the risk ladder.")

input_cols = st.columns(3)

with input_cols[0]:
    daily_stop = st.number_input(
        "Daily Stop ($)",
        min_value=0.0,
        value=150.0,
        step=10.0,
        format="%.2f",
        help="Max amount you allow yourself to lose today.",
    )

with input_cols[1]:
    stop_loss_per_share = st.number_input(
        "Stop Loss per Share ($)",
        min_value=0.0,
        value=0.25,
        step=0.01,
        format="%.2f",
        help="Distance between entry and stop.",
    )

with input_cols[2]:
    entry_price = st.number_input(
        "Entry Price - Optional ($)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.2f",
        help="Used to estimate position value for each preset.",
    )

st.subheader("Quick Risk Ladder")
st.caption("Preset-by-preset sizing from your three core inputs.")

if daily_stop > 0 and stop_loss_per_share > 0:
    ladder_rows = []

    for label, multiplier in RISK_PRESETS.items():
        risk_budget = daily_stop * multiplier
        shares = int(risk_budget / stop_loss_per_share)
        actual_risk = shares * stop_loss_per_share

        equity = f"{shares * entry_price:,.2f}" if entry_price > 0 else "-"
        row = {
            "Presets": label,
            "Actual Risk ($)": f"{actual_risk:,.2f}",
            "Shares": f"{shares:,}",
            "Equity ($)": equity,
        }

        ladder_rows.append(row)

    st.dataframe(ladder_rows, width="stretch", hide_index=True)
else:
    st.info("Enter Daily Stop and Stop Loss per Share above 0 to populate the ladder.")
