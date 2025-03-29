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
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def init_cache():
    index = faiss.IndexFlatL2(768)
    encoder = SentenceTransformer("all-mpnet-base-v2", use_auth_token='')
    return index, encoder


def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache_data = json.load(file)
        cache = {
            "questions": OrderedDict(cache_data.get("questions", {})),
            "response_text": cache_data.get("response_text", []),
            "frequencies": cache_data.get("frequencies", {})  # Load frequency counts
        }
    except FileNotFoundError:
        cache = {"questions": OrderedDict(), "response_text": [], "frequencies": {}}
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
        self.max_response = max_response
        # Map to track original addition index for response_text alignment
        self.question_addition_index = {q: i for i, q in enumerate(self.cache["questions"].keys())}
        # Rebuild FAISS index from cached questions if any exist
        if self.cache["questions"]:
            embeddings = self.encoder.encode(list(self.cache["questions"].keys()))
            self.index.add(embeddings)

    def evict(self):
        while len(self.cache["questions"]) > self.max_response:
            self.cache["questions"].popitem(last=False)
            self.cache["response_text"].pop(0)
            # Note: frequencies are not removed here, preserved for cleanup

    def ask(self, question: str, rag_system) -> str:
        start_time = time.time()
        embedding = self.encoder.encode([question])

        D, I = self.index.search(embedding, 1)
        if len(I[0]) > 0 and I[0][0] >= 0 and D[0][0] <= self.threshold:
            row_id = int(I[0][0])
            response_text = self.cache["response_text"][row_id]
            matched_question = list(self.cache["questions"].keys())[row_id]
            # Increment frequency for matched question
            self.cache["frequencies"][matched_question] = self.cache["frequencies"].get(matched_question, 0) + 1
            self.cache["questions"].move_to_end(matched_question)
            print("Response came from CACHE.")
        else:
            response_text = rag_system.run(question)
            print("Response came from VECTORSTORE/RAG.")
            # Set frequency to 1 for new question and store addition index
            self.cache["frequencies"][question] = 1
            self.question_addition_index[question] = len(self.cache["response_text"])
            self.cache["questions"][question] = None
            self.cache["response_text"].append(response_text)
            self.index.add(embedding)

        self.evict()
        store_cache(self.json_file, self.cache)
        print(f"Time taken: {time.time() - start_time:.3f} seconds")
        return response_text

    def cleanup_cache(self):
        # Filter questions with frequency >= 2, apply LRU to low-frequency ones
        low_freq_questions = [(q, self.cache["frequencies"].get(q, 0)) for q in self.cache["questions"].keys() if self.cache["frequencies"].get(q, 0) < 2]
        high_freq_questions = [(q, self.cache["frequencies"].get(q, 0)) for q in self.cache["questions"].keys() if self.cache["frequencies"].get(q, 0) >= 2]

        # Sort low-frequency questions by LRU (earlier in OrderedDict = older)
        low_freq_questions.sort(key=lambda x: list(self.cache["questions"].keys()).index(x[0]))

        # Keep all high-frequency questions, no low-frequency ones (remove all < 2)
        kept_questions = [q for q, freq in high_freq_questions]
        
        if not kept_questions:
            # If no questions meet the threshold, clear the cache
            self.cache["questions"] = OrderedDict()
            self.cache["response_text"] = []
            self.cache["frequencies"] = {}
        else:
            # Reconstruct cache with only high-frequency questions
            new_questions = OrderedDict()
            new_response_text = []
            new_frequencies = {}
            for q in kept_questions:
                new_questions[q] = None
                new_response_text.append(self.cache["response_text"][self.question_addition_index[q]])
                new_frequencies[q] = self.cache["frequencies"][q]
            self.cache["questions"] = new_questions
            self.cache["response_text"] = new_response_text
            self.cache["frequencies"] = new_frequencies
        
        # Save the filtered cache
        store_cache(self.json_file, self.cache)
        print("Cache cleaned up. Low-frequency questions (< 2) removed, frequencies preserved in cache_file.json.")


def ingest_pdfs(pdf_folder):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_folder, file))
            text = " ".join([page.get_text("text") for page in doc])
            documents.append(Document(page_content=text, metadata={"source": file}))
    return documents


def build_rag_system(pdf_folder):
    documents = ingest_pdfs(pdf_folder)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv('GOOGLE_API_MODEL'))
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv('GOOGLE_API_MODEL'), temperature=0.4, convert_system_message_to_human=True) 
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return rag_chain


if __name__ == "__main__":
    cache = SemanticCache()
    rag_system = build_rag_system("pdfdata")
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() in ["exit", "quit"]:
            cache.cleanup_cache()
            break
        response = cache.ask(user_input, rag_system)
        print(f"Response: {response}")
