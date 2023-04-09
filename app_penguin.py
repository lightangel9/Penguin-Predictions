import streamlit as st 
from streamlit_option_menu import option_menu
import plotly.graph_objects as px
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


model = pickle.load(open('model.penguins.sav','rb'))
species_encoder = pickle.load(open('encoder.species.sav','rb'))
island_encoder = pickle.load(open('encoder.island.sav','rb'))
sex_encoder = pickle.load(open('encoder.sex.sav','rb'))
evaluations = pickle.load(open('evals.all.sav','rb'))


st.set_page_config(
    page_title="Penguin",
    layout="wide"
)

st.title("Penguin Species Prediction")

'''
## Penguin .... ^.^ 
เพกวินเป็นนก ... แต่บินไม่ได้
'''

# Create two columns in the layout
c1, c2 = st.beta_columns(2)

# Add Prediction menu to column c1
with c1:
    st.subheader("Prediction")
    x1 = st.radio("เลือก island ",island_encoder.classes_)
    x1 = island_encoder.transform([x1])[0]
    x2 = st.slider("เลือก culmen length (mm)", 20,70,35 )
    x3 = st.slider("เลือก culmen depth (mm)", 10,30,15 )
    x4 = st.slider("เลือก flipper length (mm)", 150,250,200)
    x5 = st.slider("เลือก body mass (g)", 2500,6500,3000)
    x6 = st.radio("เลือก sex ",sex_encoder.classes_)
    x6 = sex_encoder.transform([x6])[0]
    x_new = pd.DataFrame(data=np.array([x1, x2, x3, x4, x5, x6]).reshape(1,-1), 
                 columns=['island', 'culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g', 'sex'])

    pred = model.predict(x_new)

    html_str = f"""
    <style>
    p.a {{
      font: bold {30}px Courier;
    }}
    </style>
    <p class="a">{species_encoder.inverse_transform(pred)[0]}</p>
    """

    st.markdown('### Predicted Species: ' )
    st.markdown(html_str, unsafe_allow_html=True)

# Add Evaluation menu to column c2
with c2:
    st.subheader("Evaluation")
    x = evaluations.columns
    fig = px.Figure(data=[
        px.Bar(name = 'Decision Tree',
               x = x,
               y = evaluations.loc['Decision Tress']),
        px.Bar(name = 'Random Forest',
               x = x,
               y =  evaluations.loc['Random Forest']),
        px.Bar(name = 'KNN',
               x = x,
               y =  evaluations.loc['KNN']),
        px.Bar(name = 'AdaBoost',
               x = x,
               y =  evaluations.loc['AdaBoost']),
        px.Bar(name = 'XGBoost',
               x = x,
               y =  evaluations.loc['XGBoost'])
    ])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(evaluations)
   
st.sidebar.info("**💾 More informations:**")
st.sidebar.caption("[🔗Github](https://github.com/lightangel9/Penguin-Predictions)")
