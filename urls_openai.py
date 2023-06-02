import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

urls = [
    'https://www.mosaicml.com/blog/mpt-7b',
    'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models', 
    'https://lmsys.org/blog/2023-03-30-vicuna/',
    'https://huggingface.co/tiiuae/falcon-40b',
    'https://huggingface.co/tiiuae/falcon-7b',
]

loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n', 
                                      chunk_size=1000, 
                                      chunk_overlap=50)

docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()

vectorStore_openAI = FAISS.from_documents(docs, embeddings)

filepath = "faiss_store_openai.pkl"
if not os.path.isfile(filepath):
    with open(filepath, "wb") as f:
        pickle.dump(vectorStore_openAI, f)
    
with open(filepath, "rb") as f:
    VectorStore = pickle.load(f)
    
llm=OpenAI(temperature=0, ) # model_name='')

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

user_input = "Tell me about the Falcon-7B model"
output = chain({"question": user_input}, return_only_outputs=True)

print(output['answer'])
print(output['sources'])

