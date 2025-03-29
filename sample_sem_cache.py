import os
import json
import time
import faiss
import fitz  # PyMuPDF for PDF processing
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import OrderedDict

# Updated community imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def init_cache():
    index = faiss.IndexFlatL2(768)
    # Optionally, set your Hugging Face token if needed:
    # os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"
    encoder = SentenceTransformer("all-mpnet-base-v2", use_auth_token='')
    return index, encoder


def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
        cache = {
            "questions": OrderedDict(cache.get("questions", {})),
            "response_text": cache.get("response_text", [])
        }
    except FileNotFoundError:
        cache = {"questions": OrderedDict(), "answers": [], "response_text": []}
    return cache


def store_cache(json_file, cache):
    with open(json_file, "w") as file:
        json.dump(cache, file)


class SemanticCache:
    def __init__(self, json_file="cache_file.json", threshold=0.35, max_response=100):
        self.index, self.encoder = init_cache()
        self.threshold = threshold
        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        if self.cache["questions"]:
            embeddings = self.encoder.encode(list(self.cache["questions"].keys()))
            self.index.add(embeddings)
        self.max_response = max_response

    def evict(self):
        while len(self.cache["questions"]) > self.max_response:
            self.cache["questions"].popitem(last=False)
            # self.cache["embeddings"].pop(0)
            # self.cache["answers"].pop(0)
            self.cache["response_text"].pop(0)

    def ask(self, question: str, rag_system) -> str:
        start_time = time.time()
        embedding = self.encoder.encode([question])

        D, I = self.index.search(embedding, 1)
        if D[0] >= 0 and I[0][0] >= 0 and D[0][0] <= self.threshold:
            row_id = int(I[0][0])
            response_text = self.cache["response_text"][row_id]
            matched_question = list(self.cache["questions"].keys())[row_id]
            self.cache["questions"].move_to_end(matched_question)
            print("Response came from CACHE.")
        else:
            response_text = rag_system.run(question)
            print("Response came from VECTORSTORE/RAG.")
            self.cache["questions"][question] = None
            # self.cache["embeddings"].append(embedding[0].tolist())
            self.cache["answers"].append(response_text)
            self.cache["response_text"].append(response_text)
            self.index.add(embedding)

        self.evict()
        store_cache(self.json_file, self.cache)
        print(f"Time taken: {time.time() - start_time:.3f} seconds")
        return response_text


def ingest_pdfs(pdf_folder):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_folder, file))
            text = " ".join([page.get_text("text") for page in doc])
            documents.append(Document(page_content=text, metadata={"source": file}))
    return documents


def build_rag_system(pdf_folder):
    # Load documents from PDFs
    documents = ingest_pdfs(pdf_folder)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv('GOOGLE_API_MODEL'))
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=os.getenv('GOOGLE_API_MODEL'), temperature=0.4, convert_system_message_to_human=True) 
    # Use the new RetrievalQA chain construction method.
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return rag_chain


if __name__ == "__main__":
    cache = SemanticCache()
    rag_system = build_rag_system("pdfdata")
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = cache.ask(user_input, rag_system)
        print(f"Response: {response}")
