from typing import Optional, Dict
import os
import chainlit as cl
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, AsyncCallbackManagerForLLMRun
from langchain_community.llms import LlamaCpp
from chainlit.types import ThreadDict
from langchain.chains import RetrievalQA, ConversationChain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory



# Initialize the language model with LlamaCpp
llm = LlamaCpp(model_path="Model/llama-2-7b-chat.Q4_K_M.gguf",  #  token streaming to terminal
               device="cpu",verbose = True, max_tokens = 2048,  #offloads ALL layers to GPU, uses around 6 GB of Vram
               config={  # max tokens in reply
                       'temperature': 0.75}  # randomness of the reply
               )

DATA_PATH = 'Data/'

DB_CHROMA_PATH = 'vectorstore/db_chroma'

embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_function)


rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=db.as_retriever(),
    return_source_documents=True
)


template = """
You are an AI specialized in the medical domain. 
Your purpose is to provide accurate, clear, and helpful responses to medical-related inquiries. 
You must avoid misinformation at all costs. Do not respond to questions outside of the medical domain. 
If you are unsure or lack information about a query, you must clearly state that you do not know the answer.

Question: {query}

Answer:

"""

prompt_template = PromptTemplate(input_variables=["query"],template=template)



conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(), 
)




@cl.on_chat_start
async def on_chat_start():
    pass




@cl.step(type="llm")
def get_response(query):
    """
    Generates a response from the language model based on the user's input. If the input includes
    '-rag', it uses a retrieval-augmented generation pipeline, otherwise, it directly invokes
    the language model.

    Args:
        question (str): The user's input text.

    Returns:
        str: The language model's response, potentially including source documents if '-rag' was used.
    """
    
   
    if "-rag" in query.lower():
        response = rag_pipeline(prompt_template.format(query=query))
        result = response["result"]
        source = response["source_documents"]
        if source:
            source_details = "\n\nSources:"
            for source in source:
                page_content = source.page_content 
                page_number = source.metadata['page']
                source_book = source.metadata['source']
                source_details += f"\n- Page {page_number} from {source_book}: \"{page_content}\""
        
            result += source_details
        return result
        
     
    return llm.invoke(prompt_template.format(query=query))



@cl.on_message
async def on_message(message: cl.Message):
    """
    Fetches the response from the language model and shows it in the web ui.
    """
    try:
        response = get_response(message.content)
        msg = cl.Message(content=response)
    except Exception as e:
        msg = cl.Message(content=str(e))

    await msg.send()




@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    pass # TODO user history gets fed to LLM 



@cl.on_chat_end
def on_chat_end():
    pass 


provider_id = os.getenv('OAUTH_GOOGLE_CLIENT_ID')
token = os.getenv('OAUTH_GOOGLE_CLIENT_SECRET')
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:

    
    # Allow any Gmail user to authenticate
    if provider_id == "google":
        email = raw_user_data.get("email", "")
        if email.endswith("@gmail.com"):
            return default_user

    return None 