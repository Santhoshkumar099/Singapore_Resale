import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import datetime
import pickle
from sklearn.tree import DecisionTreeRegressor



import numpy as np
import pickle
import streamlit as st
import pandas as pd
import json
import datetime as dt



#page config
st.set_page_config(page_title="Flat Price Prediction",page_icon="🏢")
st.sidebar.image("MDTM20\Project05\Singapore01.jfif")



# creating option menu in side bar
with st.sidebar:
    selected = option_menu("Menu", ["🏠Home","📈Predicition","❗About"],
                menu_icon= "menu-button-wide",
                default_index=0,
                styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "#B7CAFC"},
                        "nav-link-selected": {"background-color": "#B7CAFC"}})


#menu Home
if selected=="🏠Home":
    st.markdown("<h1 style='text-align: center; color: Red;'>Singapore Flat Resale Price</h1>",unsafe_allow_html=True)
    col1,col2= st.columns(2)
    with col1:
        st.subheader("About HDB")
        st.markdown("Housing and development in Singapore are notable for their strategic planning, efficient execution, and focus on creating a high standard of living for its residents.")
        st.link_button(label='Official Website',url='https://www.hdb.gov.sg/cs/infoweb/homepage',use_container_width=True)
    with col2:
        st.subheader('Tools and Technologies used')
        st.markdown(' Python, Pandas, numpy, matplotlib, seaborn, Plotly, Streamlit, sklearn')
        st.subheader('ML Model')
        st.markdown('The ML model used in this project is :blue[DecisionTree].')

    st.write("")
    col1,col2= st.columns(2)
    with col1:
        st.video("https://www.youtube.com/watch?v=T6uqnLtEWZE")
    with col2:
        st.video("https://www.youtube.com/watch?v=17BZB6ko_Nc")
    
if selected=="📈Predicition":
        with st.form('prediction'):
            col1, col2 = st.columns(2, gap='large')

            with col1:
                month_mapping= {"January" : 1,"February" : 2,"March" : 3,"April" : 4,"May" : 5,"June" : 6,"July" : 7,
                                "August" : 8,"September" : 9,"October" : 10 ,"November" : 11,"December" : 12}
                month_key=st.selectbox("**Select the Month**",list(month_mapping.keys()))
                resale_month=month_mapping[month_key]

                resale_year = st.number_input('**Enter the resale year**', value=2016)

                town_mapping = {'ANG MO KIO': 1, 'BEDOK': 2, 'BISHAN': 3, 'BUKIT BATOK': 4, 'BUKIT MERAH': 5, 'BUKIT TIMAH': 6,
                            'CENTRAL AREA': 7, 'CHOA CHU KANG': 8, 'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11,
                            'JURONG EAST': 12, 'JURONG WEST': 13, 'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'QUEENSTOWN': 16,
                            'SENGKANG': 17, 'SERANGOON': 18, 'TAMPINES': 19, 'TOA PAYOH': 20, 'WOODLANDS': 21, 'YISHUN': 22,
                            'LIM CHU KANG': 23, 'SEMBAWANG': 24, 'BUKIT PANJANG': 25, 'PASIR RIS': 26, 'PUNGGOL': 27}

                town_key = st.selectbox('**Select a town**', list(town_mapping.keys()))
                town = town_mapping[town_key]

                block=st.number_input('**Block**')


                flat_model_mapping = {'IMPROVED': 1, 'NEW GENERATION': 2, 'MODEL A': 3, 'STANDARD': 4, 'SIMPLIFIED': 5,
                                'MODEL A-MAISONETTE': 6, 'APARTMENT': 7, 'MAISONETTE': 8, 'TERRACE': 9, '2-ROOM': 10,
                                'IMPROVED-MAISONETTE': 11,
                                'MULTI GENERATION': 12, 'PREMIUM APARTMENT': 13, 'Improved': 14, 'New Generation': 15,
                                'Model A':
                                    16, 'Standard': 17, 'Apartment': 18, 'Simplified': 19, 'Model A-Maisonette': 20,
                                'Maisonette':
                                    21, 'Multi Generation': 22, 'Adjoined flat': 23, 'Premium Apartment': 24, 'Terrace': 25,
                                'Improved-Maisonette': 26, 'Premium Maisonette': 27, '2-room': 28, 'Model A2': 29, 'DBSS': 30,
                                'Type S1': 31, 'Type S2': 32, 'Premium Apartment Loft': 33, '3Gen': 34}

                flat_model_value = st.selectbox("**Select Flat Model**", list(flat_model_mapping.keys()))
                flat_model = flat_model_mapping[flat_model_value] 

            with col2:

                category_mapping = {
                '1 ROOM': 1,
                '2 ROOM': 2,
                '3 ROOM': 3,
                '4 ROOM': 4,
                '5 ROOM': 5,
                'EXECUTIVE': 6,
                'MULTI GENERATION': 7
                }

                flat_type_value = st.selectbox('**Select Flat Type**', list(category_mapping.keys()))
                flat_type = category_mapping[flat_type_value]

                floor_area_sqm= st.number_input("**Enter the area**", value=35.0)

                storey_lower_bound = st.number_input("**Enter the lower bound of the storey range**", min_value=0)

                storey_upper_bound= st.number_input("**Enter the upper bound of the storey range**", )

                lease_commence_date = st.number_input("**Enter the lease commence year**", value=1990)

                st.markdown('<br>', unsafe_allow_html=True)

                button=st.form_submit_button('PREDICT',use_container_width=True)

            if button:
                with st.spinner("Predicting..."):

                    #check whether user fill all required fields
                    if not all([resale_month,town,flat_type,flat_model,floor_area_sqm,resale_year,block,
                                lease_commence_date,storey_lower_bound,storey_upper_bound]):
                        st.error("Please fill in all required fields.")

                    else:
                        floor_area_sqm_log=np.log(floor_area_sqm)

                        #opened pickle model and predict the resale price with user data
                        with open('F:\IT Field\Python01\MDTM20\Project05\Decisiontree.pkl','rb') as files:
                            model=pickle.load(files)
                        
                        user_data=np.array([[town, flat_type, block, flat_model, lease_commence_date,resale_year,
                                            resale_month,storey_lower_bound,storey_upper_bound, floor_area_sqm_log ]])

                        predict=model.predict(user_data)
                        resale_price=np.exp(predict[0])

                        #display the predicted selling price 
                        st.markdown(f"### :blue[Flat Resale Price is] :green[$ {round(resale_price, 3)}]")

            
if selected=="❗About":

    c1,c2=st.columns(2)
    with c1:
        st.subheader("Housing Development Board (HDB): ")
        st.write("##### Established in 1960, HDB is responsible for public housing in Singapore. It has successfully housed over 80% of the population, with around 90% of residents owning their homes. HDB flats are well-designed, affordable, and cater to various income groups.")
        st.write(" ")
        st.link_button(':red[OFFICIAL WEBSITE]',url='https://www.hdb.gov.sg/cs/infoweb/homepage')
    
    with c2:
        st.image('https://media2.malaymail.com/uploads/articles/2020/2020-07/20200725_Singapore-HDB.jpg')
    
    

    st.write('')
    st.write('')
    st.header(':red[Personal Information]')
    Name = (f'{":red[Name] :"}  {"Santhosh Kumar M"}')
    mail = (f'{":red[Mail] :"}  {"sksanthoshhkumar99@gmail.com"}')
    st.markdown(Name)
    st.markdown(mail)
    c1,c2=st.columns(2)
    with c1:
        if st.button(':red[Show Github Profile]'):
            st.markdown('[Click here to visit github](https://github.com/Santhoshkumar099)')

    with c2:
        if st.button(':red[Show Linkedin Profile]'):
            st.markdown('[Click here to visit linkedin](https://www.linkedin.com/in/santhosh-kumar-2040ab188/)')

    github = (f'{"Github :"}  {"https://github.com/Santhoshkumar099"}')
    linkedin = (f'{"LinkedIn :"}  {""}')
    description = "An Aspiring DATA-SCIENTIST..!"

    st.markdown("This project is done by Santhosh Kumar M")