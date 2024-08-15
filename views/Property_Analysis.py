import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
import joblib


################################################################

# geomap


st.set_page_config(page_title="Plotting Demo")

st.title('Analytics')

new_df = pd.read_csv('data/data_viz1.csv')

group_df = new_df.groupby('sector')[['price','price_per_sqft','builtup_area','latitude','longitude']].mean()

st.header('Sector Price per Sqft Geomap')
fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='builtup_area',
                  color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                  mapbox_style="open-street-map",width=1200,height=700,hover_name=group_df.index)


st.plotly_chart(fig,use_container_width=True)

#####################################################

#wordcloud

with open('data/df_cloud.pkl','rb') as file:
    df_word = pickle.load(file)



#st.write(df_word.dropna())
st.header('Sector-wise amenities')

sectorx = st.selectbox('Select Sector', df_word['sector'].to_list())

def sec_cloud(x):
    y = ast.literal_eval(df_word[df_word['sector'] == x].iloc[0,0])
    new_y = [word.replace(' ', '').replace('-','') for word in y]
    return ' '.join(new_y)

feature = sec_cloud(sectorx)

wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='black',
                      stopwords = set(['s']),  # Any stopwords you'd like to exclude
                      min_font_size = 10).generate(feature)


##fig11 ,ax = plt.subplots()
##ax.imshow(wordcloud, interpolation='bilinear')
fig11 = plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)
st.pyplot(fig11)
##st.set_option('deprecation.showPyplotGlobalUse', False)
#plt.show()


#########################################################

#scatter chart

st.header('Area vs Price')

property_type = st.selectbox('Select Property Type', ['flat','house'])

if property_type == 'house':
    fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="builtup_area", y="price", color="bedRoom")

    st.plotly_chart(fig1, use_container_width=True)
else:
    fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="builtup_area", y="price", color="bedRoom"
                      )

    st.plotly_chart(fig1, use_container_width=True)

##########################################################

#pie chart

st.header('BHK Pie Chart')

sector_options = new_df['sector'].unique().tolist()
sector_options.insert(0,'overall')

selected_sector = st.selectbox('Select Sector', sector_options)

if selected_sector == 'overall':

    fig2 = px.pie(new_df, names='bedRoom')

    st.plotly_chart(fig2, use_container_width=True)
else:

    fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')

    st.plotly_chart(fig2, use_container_width=True)


################################################

# box plot

st.header('Side by Side BHK price comparison')

fig3 = px.box(new_df[new_df['bedRoom'] <= 5], x='bedRoom', y='price', title='BHK Price Range')

st.plotly_chart(fig3, use_container_width=True)

###################################################

#displot

st.header('Distplot for property type')

fig3 = plt.figure(figsize=(10, 4))
sns.distplot(new_df[new_df['property_type'] == 'house']['price'],label='house')
sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat')
plt.legend()
st.pyplot(fig3)
##st.set_option('deprecation.showPyplotGlobalUse', False)



