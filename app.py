import streamlit as st
import pandas as pd

def main():
 
    st.markdown("<h1 style='text-align: center;'>Stock Price Predictions </h1>", unsafe_allow_html=True)
    st.write (':money_with_wings:')
    st.sidebar.title('User Input Features') 
    
    # select algorithm
    st.sidebar.info('Welcome to the Stock Price Predictor App!')
  
    
    algorithm = st.sidebar.selectbox('Select Algorithm', ('Linear Regression', 'SVM Regressor', 'LSTM'))
    
    st.sidebar.markdown("---")
  
    st.sidebar.subheader('Lựa chọn tác vụ')
    option = st.sidebar.radio('Chọn một tab:', ['1 day', '1 week', '1 month', 'Else'])

    if option == '1 day':
        st.sidebar.write('You choose 1 day')
    elif option == '1 week':
        st.sidebar.write('You choose 1 week')
    elif option == '1 month':
        st.sidebar.write('You choose 1 month')
    else:
        st.sidebar.write('You choose custom input')
        num = st.sidebar.number_input('How many days forecast?', value=5)
        num = int(num)
        
    if st.sidebar.button('Predict') and algorithm:
        if option == '1 day':
            st.write(f'Predict for 1 day with {algorithm}')
        else: 
            st.write('Quoc bi gay')
        
    
    
    
if __name__ == '__main__':
    main()