"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os
import pexpect

# From here down is all the StreamLit UI.
st.set_page_config(page_title="📊 ChatCSV", page_icon="📊")
st.header("📊 ChatCSV")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []
    
    
from langchain.agents import load_tools, initialize_agent, AgentType, Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent

df = pd.DataFrame([])
data = st.file_uploader(label='Upload CSV file', type='csv')

# st.download_button(label='サンプルデータをダウンロードする',data='https://drive.google.com/file/d/1wuSx35y3-hjZew1XhrM78xlAGIDTd4fp/view?usp=drive_open',mime='text/csv')

header_num = st.number_input(label='Header position',value=0)
index_num = st.number_input(label='Index position',value=2)
index_list = [i for i in range(index_num)]

if data:
    df = pd.read_csv(data,header=header_num,index_col=index_list)
    st.dataframe(df)

def get_text():
    input_text = st.text_input("You: ", "Tell me the average of the revenue", key="input")
    return input_text


ask_button = ""

if df.shape[0] > 0:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0, max_tokens=1000), df, verbose=True, return_intermediate_steps=True)
    user_input = get_text()
    ask_button = st.button('ask')
else:
    pass

language = st.selectbox('language',['English','日本語'])


import json
if ask_button:
     with st.spinner('typing...'):
        chat_history = []
        prefix = f'You are the best explainer. please answer in {language}. User: '
        response = agent({"input":user_input})
        result = json.dumps(response['intermediate_steps'], indent=2, ensure_ascii=False).replace('[\n', '').replace(']\n', '').replace(']', '')
    
        st.session_state.past.append(user_input)
        st.session_state.generated.append(result)
        # st.session_state.generated.append(response['output'])
        # chat_history.append(user_input)
        # chat_history.append(result)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
