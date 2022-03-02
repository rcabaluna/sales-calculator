import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns

# Title
st.title("Sales Calculator App")
st.write("""This app calculates the possible sales based on the budget spent on different advertising platforms.""")

columns = ['TV','Radio','Newspaper']

data = pd.read_csv("Advertising.csv")
data = data.iloc[: , 1:]




# Sidebar Options
st.sidebar.title('Options')
if st.sidebar.checkbox('Toggle Dataframe'):
    st.write(data.head(20))
st.sidebar.info('This checkbox shows/hides the dataframe.')


st.sidebar.title('Parameters')


# TEXT INPUTS
xTV = st.sidebar.text_input("TV",1)
xRadio = st.sidebar.text_input("Radio",1)
xNewspaper = st.sidebar.text_input("Newspaper",1)



# Loading Model
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)
prediction = loaded_model.predict([[xTV,xRadio,xNewspaper]])[0]


st.subheader('Prediction')
st.write(f"Predicted Sales based on advertising budget: {prediction}")


### Figures
# Figure 1 
st.title("Figure 1")
st.write("""The figure 1 shows the relationship of Sales per spent on TV advertising.""")

sns.scatterplot(x='Sales', y='TV', data=data)
st.pyplot()

#Figure 2
st.title("Figure 2")
sns.pairplot(data)
st.pyplot()

#Figure 3
st.title("Figure 3")
st.line_chart(data)

