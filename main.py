"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os
import pexpect

# From here down is all the StreamLit UI.
st.set_page_config(page_title="📊 ChatCSV", page_icon="📊")
st.header("📊 ChatCSV")

df = ''
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []
    
    
from langchain.agents import load_tools, initialize_agent, AgentType, Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

data = st.file_uploader(label='Upload CSV file', type='csv')

if data:
    header_num = st.number_input(label='ヘッダーの位置',value=0)
    index_num = st.number_input(label='インデックスの位置',value=2)
    index_list = [i for i in range(index_num)]
    submit_button = st.button('submit')

if data and submit_button:
    df = pd.read_csv(data,header=header_num,index_col=index_list)
    st.dataframe(df)

def get_text():
    input_text = st.text_input("You: ", "線形回帰でこの期間のあとの5カ月の利益予測をしてください", key="input")
    return input_text


ask_button = ""

if df.shape[0] > 0:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0, max_tokens=1000), df, verbose=True)
    user_input = get_text()
    ask_button = st.button('ask')
else:
    pass

language = st.selectbox('language',['English','日本語'])

if ask_button:
    chat_history = []
    prefix = f'You are the best explainer. please answer in {language}. User: '
    result = agent.run(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result)
    # chat_history.append(user_input)
    # chat_history.append(result)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
