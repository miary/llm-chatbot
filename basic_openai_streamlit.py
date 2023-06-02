import os
import streamlit as st 
from dotenv import load_dotenv
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


#--------- UI Components ------------------#
st.sidebar.image("assets/CBP-logo.png")
st.sidebar.header("CBP Assets Query")
st.sidebar.markdown("1. Assets and models are localized.")
st.sidebar.markdown("2. Documents are within the firewall.")
st.sidebar.markdown("3. Queries don't go through the internet.")

st.sidebar.divider()
st.sidebar.markdown("Document types can be:")
st.sidebar.markdown("- MS Word, Excell")
st.sidebar.markdown("- PDF and Text documents")
st.sidebar.markdown("- HTML documents -- from sitemap or individual links")
st.sidebar.markdown("- Many more")

st.header("CBP Generative AI Knowledge Query")

#-------- Langchain LLM output displayed on UI ------#
def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
