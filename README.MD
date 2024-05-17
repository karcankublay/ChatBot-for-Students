Sample project: https://github.com/flyfir248/LLAMA-2B-Medical-Bot-using-Chainlit

#### How to run
0. Download model from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf and place it under `Model` directory
0.5 Have a running anaconda or miniconda installation (llama-cpp-python is only available at conda forge)
1. Make a new virtual environment `conda create -n <name of your environment> python = 3.10`
2. `conda activate <name of your environment>`
3. `pip install -r requirements.txt`
4. `conda install -c conda-forge llama-cpp-python`
5. `chainlit run app.py -w`

The welcome screen can be edited by changing chainlit.md

Don't forget to uncomment lines : `install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 ` IFF you want to generate vector db with CUDA acceleration

RAG example: https://github.com/samwit/langchain-tutorials/blob/main/RAG/YT_Chroma_DB_Multi_doc_retriever_Langchain_Part1.ipynb