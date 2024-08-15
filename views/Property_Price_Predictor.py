import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
import category_encoders as ce
import joblib
import gzip
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

#st.set_page_config(page_title="Viz Demo")


st.write("# Price Predictor Real Estate")



with open('data/df_x.pkl','rb') as file:
    df = pickle.load(file)

#with open('data/pipeline_xy.pkl','rb') as file:
   # pipeline = pickle.load(file)

#########################################################################
new_df = pd.read_csv('data/missing_value_impute_gurgaon_real_estate.csv').drop_duplicates()

#new_df = pd.read_csv('C:\\Capstone_project_real_estate\\price_file.csv',sep=',')
# categorising luxury_score feature

def cat_luxury(x):
  if 0 <= x < 30:
    return 'Low'
  elif 30 <= x < 110:
    return 'Medium'
  elif 110<= x < 175:
    return 'High'
  else:
    return None

new_df['luxury_category'] = new_df['luxury_score'].apply(cat_luxury)

# categorising floorNum feature

def cat_floor(x):
  if 0 <= x < 3:
    return 'Low Floor'
  elif 3 <= x < 11:
    return 'Mid Floor'
  elif 11<= x < 52:
    return 'High Floor'
  else:
    return None


new_df['floor_category'] = new_df['floorNum'].apply(cat_floor)

new_df.drop(columns=['floorNum','luxury_score'],inplace=True)

new_df.drop(columns=['pooja room', 'study room', 'store room', 'others'],inplace=True)

new_df.drop(columns=['society','price_per_sqft'],inplace=True)

new_df['furnishing_type'] = new_df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})

#st.dataframe(new_df)


X = new_df.drop(columns=['price'],axis=0)
y = new_df['price']

# applying log transformation to target feature
y_transformed = np.log1p(y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'builtup_area', 'servant room']),
        ('target_enc',ce.TargetEncoder(),['sector']),
        ('cat1',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),['property_type','balcony','agePossession','furnishing_type','luxury_category','floor_category'])

    ],
    remainder='passthrough'
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300,max_depth=20))
])


pipeline.fit(X,y_transformed)




##########################################################################


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
    base_price = np.expm1(pipeline.predict(pd.DataFrame(one_df)))[0]
    
    low = base_price - 0.10
    high = base_price + 0.10

    # display

    if low < 1 and high < 1:
        st.text("The price of the flat/house is between {} Lac and {} Lac".format(round(low,2)*100,round(high,2)*100))
    elif low < 1 and high > 1:
        st.text("The price of the flat/house is between {} Lac and {} Cr".format(round(low,2)*100,round(high,2)))
    else:
        st.text("The price of the flat/house is between {} Cr and {} Cr".format(round(low,2),round(high,2)))
