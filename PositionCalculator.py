
import streamlit as st

st.title("Position Size Calculator")

# Percent multipliers
percent_multiplier = {
    "1% (D)": 0.01,
    "5% (C)": 0.05,
    "7.5% (C+)": 0.075,
    "10% (B-)": 0.10,
    "12.5% (B)": 0.125,
    "15% (B+)": 0.15,
    "30% (A-)": 0.30,
    "50% (A)": 0.50,
    "80% (A+)": 0.80,
    "90% (A++)": 0.90,
    "100% (A+++)": 1,

}

# Inputs
daily_stop = st.number_input("Daily Stop ($):", placeholder="")

try:
    daily_stop = float(daily_stop) if daily_stop !="" else 0.00
except ValueError:
    daily_stop = 0.00


percent_of_DS = st.selectbox("Percent of DS:", ["-"] + list (percent_multiplier.keys()))
stop_loss_share = st.number_input("Stop Loss per Share ($):", min_value=0.0, step=0.01, format="%.2f", value=0.00)


# Calculation
if daily_stop > 0 and stop_loss_share > 0 and percent_of_DS != "-":
    allotted_risk = daily_stop * percent_multiplier[percent_of_DS]
    position_size = allotted_risk / stop_loss_share

    st.subheader(f"Position Size: {int(position_size)} shares " + (f"(${allotted_risk:.2f} risk)"))
else:
    st.info("Enter: Daily stop, Percent, and Stop loss/share to calculate position size")
