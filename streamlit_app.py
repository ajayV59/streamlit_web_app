import streamlit as st

##st.title("hello world")

# --- PAGE SETUP ---
Home = st.Page(
    "views/Home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)


Analysis = st.Page(
    "views/Property_Analysis.py",
    title="Analytics",
    icon=":material/monitoring:",
    #default=True,
)
Property_Price_Predictor = st.Page(
    "views/Property_Price_Predictor.py",
    title="Prediction",
    icon=":material/query_stats:",
)
Property_Recommendation = st.Page(
    "views/Property_Recommendation.py",
    title="Recommender",
    icon=":material/smart_toy:",
)



# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
pg = st.navigation(pages=[Home, Analysis, Property_Price_Predictor, Property_Recommendation])

pg.run()