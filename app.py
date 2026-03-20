import streamlit as st
import langchain_helper

st.title('Restrarant name generator')

cuisine=st.sidebar.selectbox("Pick a Cuisine",("Indian","Italian","Mexican","Arabic",'American'))

if cuisine:
    response=langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())
    menu_items=response['menu_items'].strip().split(",")
    st.header(f"Suggested menu items for your {response['cuisine']} restaurant are:\n")
    for item in menu_items:
        st.write(f'- {item}')






