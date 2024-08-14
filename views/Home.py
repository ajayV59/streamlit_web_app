import streamlit  as st
import category_encoders

st.set_page_config(
     page_title='Gurgaon Real Estate Analytics app',
     page_icon="https://images.app.goo.gl/1EnMBpf2A5ziDy6p9"

)

st.write("# Welcome to our application!! \U0001F642")

st.write("""1. **Property Price Predictor:**
         Welcome to our Property Price Predictor!
         Are you looking to buy or sell a property but uncertain about its market value? Our advanced machine learning model analyzes various factors such as location, property size, amenities, and recent market trends to predict accurate property prices. Whether you're a homeowner, investor, or real estate agent, our tool provides valuable insights to help you make informed decisions.""")

st.write("""2. **Analysis of Property:**
   Explore the intricate details of any property with our Analysis of Property tool. From historical sales data to neighborhood demographics, we offer comprehensive analysis to assist you in understanding the potential of a property. Whether you're researching for investment purposes or evaluating your dream home, our platform provides in-depth insights and visualizations to guide your decision-making process effectively. """)
#st.sidebar.success('Select a demo above')

st.write("""3. **Property Recommender:**
   Discover your perfect property match with our Property Recommender tool! Using state-of-the-art recommendation algorithms, we analyze your preferences, budget, location preferences, and lifestyle factors to suggest properties that best suit your needs. Whether you're searching for a cozy apartment, a spacious family home, or a lucrative investment opportunity, our personalized recommendations streamline your property search and simplify your decision-making process. Start exploring today and find your ideal property match effortlessly! """)