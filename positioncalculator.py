import streamlit as st

# Title
st.title("Position Size Calculator")

# Grades
idea_grade_multiplier = {
    "A+ (80%)": .8,
    "A (50%)": 0.50,
    "B+ (15%)": 0.15,
    "B (10%)": 0.10,
    "C (5%)": 0.05,
    "D (1%)": 0.01,
}

# User Inputs
daily_stop = (st.number_input("Daily Stop ($):", min_value=0, step=50))
grade = st.selectbox("Grade:", list(idea_grade_multiplier.keys()))
stop_loss_share = st.number_input("Stop Loss per Share ($):", min_value=0.0, step=0.01)


# Calculations
if daily_stop > 0 and stop_loss_share > 0:
    allotted_risk = daily_stop * idea_grade_multiplier[grade]
    position_size = allotted_risk / stop_loss_share

    # Results
    st.subheader(f"Position Size: {int(position_size)} shares")
    st.write(f"{idea_grade_multiplier[grade]*100:.0f}% of Daily Stop (${allotted_risk:.2f})")

else:
    st.info("Enter Daily Stop, Grade, and Stop Loss/Share to calculate position size")
