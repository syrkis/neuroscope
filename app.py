# streamlit_app.py
#     neuroscope project
# by: Noah Syrkis

# imports
import streamlit as st

st.title("Neuroscope")
st.write("Welcome to Neuroscope, a web app that allows you to visualize your brain's activity in real time. To get started, please select a dataset from the sidebar.")


# sidebar
st.sidebar.title("Select a subject")
dataset = st.sidebar.selectbox("Select a dataset", ("Subject 1", "Subject 2", "Subject 3"))
st.sidebar.write("You selected: ", dataset)

# main page
st.write("You selected: ", dataset)

# nilearn plot
st.write("Here is a plot of the brain's activity over time.")

# input form for user
st.write("Please enter your name and email address below to receive a copy of your results.")
name = st.text_input("Name")
email = st.text_input("Email")

# submit button
if st.button("Submit"):
    st.write("Thank you for using Neuroscope. Your results will be emailed to you shortly.")
    


# footer
st.write("Made with ❤️ by Noah Syrkis")






