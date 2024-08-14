import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
import category_encoders

#st.set_page_config(page_title="Viz Demo")


st.write("# Price Predictor Real Estate")



with open('C:\\Capstone_project_real_estate\\df.pkl','rb') as file:
    df = pickle.load(file)

with open('C:\\Capstone_project_real_estate\\pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)


st.header('Enter your inputs')

# property_type
property_type = st.selectbox('Property Type',['flat','house'])

# sector
sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))

bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))

built_up_area = float(st.number_input('Built Up Area'))

servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))


furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

if st.button('Predict'):

    # form a dataframe
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'builtup_area', 'servant room','furnishing_type', 
               'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    #st.dataframe(one_df)

    # predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 0.10
    high = base_price + 0.10

    # display

    if low < 1 and high < 1:
        st.text("The price of the flat/house is between {} Lac and {} Lac".format(round(low,2)*100,round(high,2)*100))
    elif low < 1 and high > 1:
        st.text("The price of the flat/house is between {} Lac and {} Cr".format(round(low,2)*100,round(high,2)))
    else:
        st.text("The price of the flat/house is between {} Cr and {} Cr".format(round(low,2),round(high,2)))