import streamlit as st

# Title
st.title("Position Size Calculator")

# Grades
idea_grade_multiplier = {
    "A+": 1,
    "A": 0.50,
    "B+": 0.15,
    "B": 0.10,
    "C": 0.05,
    "D": 0.01,
}

# User Inputs
daily_stop = st.number_input("Daily Stop ($):", min_value=0, step=50)
grade = st.selectbox("Grade:", list(idea_grade_multiplier.keys()))
stop_loss_share = st.number_input("Stop Loss per Share ($):", min_value=0.0, step=0.01)


# Calculations
if daily_stop > 0 and stop_loss_share > 0:
    allotted_risk = daily_stop * idea_grade_multiplier[grade]
    position_size = allotted_risk / stop_loss_share

    # Results
    st.subheader("Position Size")
    st.write(f"Grade: {idea_grade_multiplier[grade]*100:.0f}% of Daily Stop")
    st.write(f"Dollar Risk: ${allotted_risk:.2f}")
    st.write(f"Position Size: {int(position_size)} shares")

else:
    st.info("Enter Daily Stop, Grade, and Stop Loss/Share to calculate position size")
