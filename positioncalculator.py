
import streamlit as st

st.title("Position Size Calculator")

# Grade multipliers
grade_multiplier = {
    "A+ (80%)": 0.80,
    "A (50%)": 0.50,
    "B+ (15%)": 0.15,
    "B (10%)": 0.10,
    "C (5%)": 0.05,
    "D (1%)": 0.01,
}

# Inputs
daily_stop = st.number_input("Daily Stop ($):", placeholder="")

try:
    daily_stop = float(daily_stop) if daily_stop !="" else 0.00
except ValueError:
    daily_stop = 0.00


grade = st.selectbox("Grade:", ["-"] + list (grade_multiplier.keys()))
stop_loss_share = st.number_input("Stop Loss per Share ($):", min_value=0.0, step=0.01, format="%.2f", value=0.00)


# Calculation
if daily_stop > 0 and stop_loss_share > 0 and grade != "-":
    allotted_risk = daily_stop * grade_multiplier[grade]
    position_size = allotted_risk / stop_loss_share

    st.subheader(f"Position Size: {int(position_size)} shares")
    st.write(f"{grade_multiplier[grade]*100:.0f}% of Daily Stop (${allotted_risk:.2f})")
else:
    st.info("Enter: Daily stop, grade, and stop loss/share to calculate position size")
