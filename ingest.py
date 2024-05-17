#%%
from langchain_chroma import Chroma
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm  # Import tqdm
import chromadb


DATA_PATH = 'Data/'
DB_CHROMA_PATH = 'vectorstore/db_chroma'
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
# Create vector database
def create_vector_db():
    # Load documents
    tqdm.write("Loading PDF files...")
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    tqdm.write(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = []
    tqdm.write("Splitting documents into text chunks...")
    for doc in tqdm(documents, desc="Processing documents"):
        texts.extend(text_splitter.split_documents([doc]))  # Corrected here
    tqdm.write(f"Generated {len(texts)} text chunks from all documents.")

    # Create the embedding function using a Hugging Face model
    tqdm.write("Initializing embeddings model...")
    

    # Load documents into Chroma
    tqdm.write("Creating embeddings and building database...")
    db = Chroma.from_documents(texts, embedding_function, persist_directory=DB_CHROMA_PATH)
    tqdm.write("Database creation complete.")
    db = chromadb.PersistentClient(path=DB_CHROMA_PATH)
    return db 
db = create_vector_db()
#%%
from langchain.vectorstores import Chroma

from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_function)
llm = LlamaCpp(model_path="Model/llama-2-7b-chat.Q4_K_M.gguf",callback_manager= CallbackManager([StreamingStdOutCallbackHandler()]),  #  token streaming to terminal
               device="cuda",n_gpu_layers=-1,verbose = True, #offloads ALL layers to GPU, uses around 6 GB of Vram
               config={  # max tokens in reply
                       'temperature': 0.75}  # randomness of the reply
               )
from langchain.chains import RetrievalQA

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=db.as_retriever(),
    return_source_documents=True
)

rag_pipeline("what is Thrombocytopenia")
# %%
