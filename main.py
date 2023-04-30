"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os
import pexpect

from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# for question generator
# from langchain.chains import LLMChain
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
# from langchain.chains.question_answering import load_qa_chain

def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def load_chain(urls):
    """Logic for loading the chain you want to use should go here."""
    if is_gpt4:
        model = "gpt-4"
    else:
        model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.9, model_name=model, streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True)
    # question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    # doc_chain = load_qa_chain(llm, chain_type="map_reduce")
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,get_chat_history=get_chat_history) #,memory=ConversationBufferWindowMemory(k=10)) #,question_generator=question_generator,combine_docs_chain=doc_chain)
    return chain

def get_text():
    input_text = st.text_input("You: ", "what is this about?", key="input")
    return input_text

# From here down is all the StreamLit UI.
st.set_page_config(page_title="🔗 ChatURLs", page_icon="🔗")
st.header("🔗 ChatURLs")

is_gpt4 = st.checkbox('Enable GPT4',help="With this it might get slower")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

st.write('You can send two or more urls. Please join them with ",". i.e. www.example.com, www.example2.com')
urls = st.text_input('urls')
ask_button = ""

if urls:
    qa = load_chain(urls.split(","))
    user_input = get_text()
    ask_button = st.button('ask')
else:
    pass

language = st.selectbox('language',['English','日本語','Estonian'])

if ask_button:
    chat_history = []
    prefix = f'You are the best explainer. please answer in {language}. User: '
    result = qa({"question": prefix + user_input, "chat_history": chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    # chat_history.append(user_input)
    # chat_history.append(result)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
