from flask import Flask, request, jsonify, render_template
import os
import json
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader

app = Flask(__name__)

# Initialize LLM and other components as in the original code
local_llm = "neural-chat-7b-v3-1.Q4_K_M.gguf"
config = {
    # configuration settings as in the original code
}

llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="avx2",
    **config
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

load_vector_store = Chroma(persist_directory="stores/pet_cosine", embedding_function=embeddings)

retriever = load_vector_store.as_retriever(search_kwargs={"k":1})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form.get('query')
    # Your logic to handle the query
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = {"answer": answer, "source_document": source_document, "doc": doc}
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
