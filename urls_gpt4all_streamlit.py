import os
from dotenv import load_dotenv
import pickle
import faiss

import streamlit as st 
from streamlit_chat import message

from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import ConversationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GPT4All



#--------- UI Components ------------------#
st.set_page_config(
        page_title="CBP - LLM",
        page_icon=":robot:",
)

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

st.header("CTO Office - LLM Chatbot")
 
load_dotenv()

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

urls = [
    'https://www.cbp.gov/employee-resources/benefits/fsafeds', # fsafeds
    'https://www.cbp.gov/employee-resources/benefits/survivor-benefits', # feglic, samba ebf
    'https://www.cbp.gov/employee-resources/benefits/tap', # samba tap
    'https://www.cbp.gov/trade/rulings/eruling-requirements', # eruling template
    'https://www.cbp.gov/trade/forced-labor', # tariff act of 1930
]

# loaders = UnstructuredURLLoader(urls=urls)
# data = loaders.load()

# # Text Splitter
# text_splitter = CharacterTextSplitter(separator='\n', 
#                                       chunk_size=1000, 
#                                       chunk_overlap=50)

# docs = text_splitter.split_documents(data)

# embeddings = OpenAIEmbeddings()

# vectorStore_openAI = FAISS.from_documents(docs, embeddings)

filepath = "faiss_store_openai.pkl"
# if not os.path.isfile(filepath):
#     with open(filepath, "wb") as f:
#         pickle.dump(vectorStore_openAI, f)

with open(filepath, "rb") as f:
    VectorStore = pickle.load(f)

callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        
#chain = ConversationChain(llm=llm)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

user_input = st.text_input("", "Hello, how are you doing?", key="input")

if user_input:
    output = chain({"question": user_input}, return_only_outputs=True)
    src = output['sources']
    answ = output['answer'] if len(output['answer']) > 5 else " "
    if len(src) > 5 and src is not None:
        answer_sources = answ + "\n" + src
    else:
        answer_sources = answ


st.session_state.past.append(user_input)
st.session_state.generated.append(answer_sources)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style="icons")
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="lorelei-neutral")
