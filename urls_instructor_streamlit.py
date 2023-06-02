import os
from dotenv import load_dotenv
import pickle
import faiss

import streamlit as st 
from streamlit_chat import message

from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 

from langchain import OpenAI
from langchain.llms import GPT4All

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# OpenAI Embedding
from langchain.embeddings import OpenAIEmbeddings

# InstructorEmbedding 
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings


#--------- UI Components ------------------#
st.set_page_config(
        page_title="CBP LLM",
        page_icon=":robot:",
)

st.sidebar.image("assets/CBP-logo.png")
st.sidebar.header("CBP Assets Query")
st.sidebar.markdown("1. Assets and models are localized.")
st.sidebar.markdown("2. Queried documents are within the CBP firewall.")
st.sidebar.markdown("3. Searches are performed within the CBP firewall.")

st.sidebar.divider()

st.sidebar.markdown("Document types can be:")
st.sidebar.markdown("- MS Word, Excell")
st.sidebar.markdown("- PDF and Text documents")
st.sidebar.markdown("- HTML documents -- from sitemap or individual links")
st.sidebar.markdown("- Many more")

st.header("CBP Generative AI Knowledge Query")
 
load_dotenv()

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
callbacks = [StreamingStdOutCallbackHandler()]


def pdf_dir_loader():
    pdf_loader = DirectoryLoader(f'Documents/pdf/', glob="./*.pdf", loader_cls=PyPDFLoader)
    loaded_pdf = pdf_loader.load()
    return loaded_pdf


def url_loader():
    urls = [
        'https://www.cbp.gov/employee-resources/benefits/fsafeds', # fsafeds
        'https://www.cbp.gov/employee-resources/benefits/survivor-benefits', # feglic, samba ebf
        'https://www.cbp.gov/employee-resources/benefits/tap', # samba tap
        'https://www.cbp.gov/trade/rulings/eruling-requirements', # eruling template
        'https://www.cbp.gov/trade/forced-labor', # tariff act of 1930
    ]

    url_loaders = UnstructuredURLLoader(urls=urls)
    url_data = url_loaders.load()
    return url_data


def text_splitting(docs_from_loader):
    text_splitter = CharacterTextSplitter(separator='\n', 
                                        chunk_size=1000, 
                                        chunk_overlap=50)
    splitted = text_splitter.split_documents(docs_from_loader)
    return splitted


def recurside_text_splitting(docs_from_text_loader):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap = 0
    )
    chunks = text_splitter.split_documents(docs_from_text_loader)
    # print summary(chunks)
    # chain.run(chunks)
    return chunks


def doc_embedding(type):
    match type:
        case 'OpenAI':
            embeddings = OpenAIEmbeddings()
        case 'Instructor':
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})
        case 'SentenceT':
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def vector_storing(docs, embeddings, storage_type):
    match storage_type:
        case 'faiss':
            vStoring = FAISS.from_documents(docs, embeddings)
        case 'chromadb':
            pass
    return vStoring

filepath = "faiss_store_openai.pkl"
#filepath = "faiss_stored_vector.pkl"
def save_load_vectorstore(filepath, vStore):
    if not os.path.isfile(filepath):
        with open(filepath, "wb") as f:
            pickle.dump(vStore, f)

    with open(filepath, "rb") as f:
        VectorStore = pickle.load(f)

    return VectorStore


def initializeLLM(llmType, storedVector):
    match llmType:
        case 'OpenAI':
            llm=OpenAI(temperature=0, 
                       model="text-davinci-003",
                       top_p=1,
                       stop=["\n"]
                       )
        case 'GPT4All':
                    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        
    #chain = ConversationChain(llm=llm)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=storedVector.as_retriever())
    return chain


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if not os.path.isfile(filepath):
    #loaded_pdf = pdf_dir_loader()
    loaded = url_loader()
    docs = text_splitting(loaded)
    #embedding_type = 'Instructor'
    embedding_type = 'SentenceT'
    embedding = doc_embedding(embedding_type)
    vStoring = vector_storing(docs, embedding, 'faiss')

storedVector = save_load_vectorstore(filepath, "")

llmType = 'GPT4All' # Options: 'OpenAI'
chain = initializeLLM(llmType, storedVector)

user_input = st.text_input("Question related to CBP: ", "Hello, how are you doing?", key="input")

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
