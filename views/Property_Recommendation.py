import streamlit as st
import pickle
import pandas as pd
import numpy as np


st.set_page_config(page_title="Recommend Apartments")

with open('C:\\Capstone_project_real_estate\\cosine_sim1.pkl','rb') as file:
    cosine_sim1 = pickle.load(file)

with open('C:\\Capstone_project_real_estate\\cosine_sim2.pkl','rb') as file:
    cosine_sim2 = pickle.load(file)

with open('C:\\Capstone_project_real_estate\\cosine_sim3.pkl','rb') as file:
    cosine_sim3 = pickle.load(file)

with open('C:\\Capstone_project_real_estate\\loc_distance.pkl','rb') as file:
    location_df = pickle.load(file)

def recommend_properties_with_scores(property_name, top_n=247):
    
    cosine_sim_matrix = 2*cosine_sim1 + 4*cosine_sim2 + 4*cosine_sim3
    # cosine_sim_matrix = cosine_sim3
    
    # Get the similarity scores for the property using its name as the index
    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
    
    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
    
    # Retrieve the names of the top properties using the indices
    top_properties = location_df.index[top_indices].tolist()
    
    # Create a dataframe with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    
    return recommendations_df

# Test the recommender function using a property name
#recommend_properties_with_scores('DLF The Camellias') 


#st.dataframe(location_df)

st.title('Select location and Radius')

selected_location = st.selectbox('Location', location_df.columns.tolist())

radi = st.number_input('Radius in km')

if st.button('Search'):
        result = location_df[location_df[selected_location] <= radi*1000][selected_location].sort_values()
        if result.empty:
           st.text('No Property found')
        else:
            for key,value in result.items():
               st.text(str(key) + " : " + str(round((value)/1000))+' KM')


st.title('Recommend Property')
selected_apartment = st.selectbox('Select Property',sorted(location_df.index.to_list()))

if st.button('Recommend'):
    recommendation_df = recommend_properties_with_scores(selected_apartment)

    st.dataframe(recommendation_df.iloc[:,0].head())
