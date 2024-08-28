import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

def main():
    st.title('Fake News Prediction')
    run_ml_app()
    
if __name__ == '__main__':
    main()