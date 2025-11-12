
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# import chromadb
# from sentence_transformers import SentenceTransformer
# from google import genai
# from google.genai import types

# # ------------------- Load API key -------------------
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# # ------------------- Initialize Gemini client -------------------
# client = genai.Client(api_key=GOOGLE_API_KEY)

# # ------------------- File & DB configuration -------------------
# PDF_FILE = "Artificial-Intelligence-in-Healthcare.pdf"
# CHROMA_PATH = "./chroma_db"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# # ------------------- Embedding model -------------------
# embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# # ------------------- ChromaDB setup -------------------
# os.makedirs(CHROMA_PATH, exist_ok=True)
# db_client = chromadb.PersistentClient(path=CHROMA_PATH)
# collection = db_client.get_or_create_collection("rag_docs")

# # ------------------- PDF chunking -------------------
# def load_pdf_chunks(pdf_path, chunk_size=400, chunk_overlap=30):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += (page.extract_text() or "") + " "
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - chunk_overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         if chunk:
#             chunks.append(chunk)
#     return chunks

# # ------------------- Embedding and store in Chroma -------------------
# def embed_and_store(chunks):
#     print(f"Embedding {len(chunks)} chunks …")
#     embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=16)
#     ids = [f"doc_{i}" for i in range(len(chunks))]
#     collection.delete()  # clear previous entries
#     collection.add(
#         embeddings=embeddings.tolist(),
#         documents=chunks,
#         ids=ids,
#     )

# # ------------------- Initialize DB -------------------
# def initialize_db():
#     if collection.count() == 0:
#         print("Indexing PDF …")
#         chunks = load_pdf_chunks(PDF_FILE)
#         embed_and_store(chunks)
#         print("Indexing done.")
#     else:
#         print("Already indexed.")

# # ------------------- Retrieve relevant chunks -------------------
# def retrieve_relevant_chunks(query, top_k=3):
#     query_embedding = embedder.encode([query])[0]
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=top_k,
#         include=["documents"],
#     )
#     return results["documents"][0]

# # ------------------- Generate answer using Gemini 2.0 -------------------
# def answer_node(query, relevant_chunks, max_output_tokens=500):
#     context = "\n".join(relevant_chunks)
#     prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"

#     try:
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",  # Using Gemini 2.0
#             contents=prompt,
#             config=types.GenerateContentConfig(
#                 max_output_tokens=max_output_tokens,
#                 temperature=0.7
#             )
#         )
#         return response.text
#     except Exception as e:
#         return f"Error during generation: {str(e)}"

# # ------------------- Simple relevance reflection -------------------
# def reflect_node(query, answer):
#     key_terms = [w.lower() for w in query.split() if len(w) > 3]
#     found = any(term in answer.lower() for term in key_terms)
#     return "✅ Relevant" if found else "⚠️ Possibly Irrelevant"

# # ------------------- Main Loop -------------------
# def main():
#     initialize_db()
#     print("\nRAG Terminal Q&A\nType your question about the PDF (or type 'exit' to quit):")
#     while True:
#         query = input("\nYour question: ").strip()
#         if query.lower() in ['exit', 'quit', 'bye']:
#             print("Goodbye!")
#             break
#         if not query:
#             print("Please enter a question.")
#             continue
#         chunks = retrieve_relevant_chunks(query)
#         answer = answer_node(query, chunks)
#         reflection = reflect_node(query, answer)
#         print(f"\nAnswer:\n{answer}\n\nReflection: {reflection}")

# # ------------------- Entry point -------------------
# if __name__ == "__main__":
#     main()
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# import chromadb
# from sentence_transformers import SentenceTransformer
# from google import genai
# from google.genai import types

# # ------------------- Load API key -------------------
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# # ------------------- Initialize Gemini client -------------------
# client = genai.Client(api_key=GOOGLE_API_KEY)

# # ------------------- File & DB configuration -------------------
# PDF_FILE = "Artificial-Intelligence-in-Healthcare.pdf"
# CHROMA_PATH = "./chroma_db"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# # ------------------- Embedding model -------------------
# embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# # ------------------- ChromaDB setup -------------------
# os.makedirs(CHROMA_PATH, exist_ok=True)
# db_client = chromadb.PersistentClient(path=CHROMA_PATH)
# collection = db_client.get_or_create_collection("rag_docs")

# # ------------------- PDF chunking -------------------
# def load_pdf_chunks(pdf_path, chunk_size=400, chunk_overlap=30):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += (page.extract_text() or "") + " "
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - chunk_overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         if chunk:
#             chunks.append(chunk)
#     return chunks

# # ------------------- Embedding and store in Chroma -------------------
# def embed_and_store(chunks):
#     print(f"Embedding {len(chunks)} chunks …")
#     embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=16)
#     ids = [f"doc_{i}" for i in range(len(chunks))]
#     collection.delete()  # clear previous entries
#     collection.add(
#         embeddings=embeddings.tolist(),
#         documents=chunks,
#         ids=ids,
#     )

# # ------------------- Initialize DB -------------------
# def initialize_db():
#     if collection.count() == 0:
#         print("Indexing PDF …")
#         chunks = load_pdf_chunks(PDF_FILE)
#         embed_and_store(chunks)
#         print("Indexing done.")
#     else:
#         print("Already indexed.")

# # ------------------- LangGraph-style nodes -------------------

# # --- Plan node ---
# # ------------------- Plan Node -------------------
# def plan_node(query):
#     """
#     Decide what to do with the query:
#     - If it's a casual greeting or very short, handle directly.
#     - Otherwise, proceed with retrieval.
#     """
#     short_queries = {
#         "hi": "Hello! How can I help you today?",
#         "hello": "Hi there! What would you like to know?",
#         "hey": "Hey! Ask me anything about the PDF.",
#         "bye": "Goodbye! Have a nice day.",
#         "thanks": "You’re welcome!",
#         "thank you": "No problem!"
#     }

#     # Normalize query
#     normalized_query = query.strip().lower()

#     # If query matches a short/casual response
#     if normalized_query in short_queries:
#         return "short_query", short_queries[normalized_query]

#     # If query is very short (<3 words), we can handle specially
#     if len(normalized_query.split()) <= 2:
#         return "short_query", f"I see you said '{query}'. Could you please elaborate?"

#     # Otherwise, proceed with RAG retrieval
#     return "rag", None


# # --- Answer node ---
# def answer_node(query, relevant_chunks, max_output_tokens=500):
#     """Generate answer using Gemini 2.0-flash"""
#     context = "\n".join(relevant_chunks)
#     prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
#     try:
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=prompt,
#             config=types.GenerateContentConfig(
#                 max_output_tokens=max_output_tokens,
#                 temperature=0.7
#             )
#         )
#         return response.text
#     except Exception as e:
#         return f"Error during generation: {str(e)}"

# # --- Reflect node ---
# def reflect_node(query, answer):
#     """Check if key terms from query exist in answer."""
#     key_terms = [w.lower() for w in query.split() if len(w) > 3]
#     found = any(term in answer.lower() for term in key_terms)
#     return "✅ Relevant" if found else "⚠️ Possibly Irrelevant"

# # ------------------- Main loop -------------------
# def main():
#     initialize_db()
#     print("\n📄 AI PDF Chat Agent (type 'exit' to quit)\n")
#     while True:
#         query = input("User: ").strip()
#         if query.lower() in ['exit', 'quit', 'bye']:
#             print("Bot: Goodbye! 👋")
#             break
#         if not query:
#             print("Bot: Please enter a question.")
#             continue

#         # Plan node to handle short queries
#         plan_result, short_response = plan_node(query)
#         if plan_result == "short_query":
#             print(f"Bot: {short_response}\n")
#             continue

#         # Normal RAG workflow
#         chunks = retrieve_relevant_chunks(query)
#         answer = answer_node(query, chunks)
#         print(f"Bot: {answer}\n")


# # ------------------- Entry point -------------------
# if __name__ == "__main__":
#     main()
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ------------------- Load API key -------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# ------------------- Initialize Gemini client -------------------
client = genai.Client(api_key=GOOGLE_API_KEY)

# ------------------- File & DB configuration -------------------
PDF_FILE = "Artificial-Intelligence-in-Healthcare.pdf"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------- Embedding model -------------------
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ------------------- ChromaDB setup -------------------
os.makedirs(CHROMA_PATH, exist_ok=True)
db_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = db_client.get_or_create_collection("rag_docs")

# ------------------- Logging setup -------------------
logs = []  # Stores trace of plan, retrieve, answer, reflect steps

def log_step(step_name, content):
    log_entry = {"step": step_name, "content": content}
    logs.append(log_entry)
    print(f"[{step_name.upper()}]: {content}")

# ------------------- PDF chunking -------------------
def load_pdf_chunks(pdf_path, chunk_size=400, chunk_overlap=30):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + " "
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# ------------------- Embedding and store in Chroma -------------------
def embed_and_store(chunks):
    log_step("embed_and_store", f"Embedding {len(chunks)} chunks …")
    embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=16)
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.delete()  # clear previous entries
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=ids,
    )

# ------------------- Initialize DB -------------------
def initialize_db():
    if collection.count() == 0:
        log_step("initialize_db", "Indexing PDF …")
        chunks = load_pdf_chunks(PDF_FILE)
        embed_and_store(chunks)
        log_step("initialize_db", "Indexing done.")
    else:
        log_step("initialize_db", "Already indexed.")

# ------------------- Retrieve relevant chunks -------------------
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents"],
    )
    chunks = results["documents"][0]
    log_step("retrieve", f"Retrieved {len(chunks)} chunks for query: '{query}'")
    return chunks

# ------------------- Generate answer using Gemini 2.0 -------------------
def answer_node(query, relevant_chunks, max_output_tokens=500):
    context = "\n".join(relevant_chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    log_step("plan", f"Preparing prompt for answer generation")

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.7
            )
        )
        answer = response.text
        log_step("answer", answer)
        return answer
    except Exception as e:
        log_step("answer", f"Error: {str(e)}")
        return f"Error during generation: {str(e)}"

# ------------------- Simple relevance reflection -------------------
def reflect_node(query, answer):
    key_terms = [w.lower() for w in query.split() if len(w) > 3]
    found = any(term in answer.lower() for term in key_terms)
    reflection = "✅ Relevant" if found else "⚠️ Possibly Irrelevant"
    log_step("reflect", reflection)
    return reflection

# ------------------- Automatic evaluation -------------------
def evaluate_answer(query, answer):
    """Basic automated evaluation: % of key terms from query found in answer"""
    query_terms = set([w.lower() for w in query.split() if len(w) > 3])
    answer_terms = set([w.lower() for w in answer.split()])
    if not query_terms:
        return 0.0
    score = len(query_terms & answer_terms) / len(query_terms)
    log_step("evaluation", f"Evaluation score: {score:.2f}")
    return score

# ------------------- Main Loop -------------------
def main():
    initialize_db()
    print("\nRAG Chat Agent (Type 'exit' to quit)")
    while True:
        user_query = input("\nUser: ").strip()
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        if not user_query:
            print("Bot: Please enter a question.")
            continue

        # ------------------- Handle short greetings -------------------
        if user_query.lower() in ['hi', 'hello', 'hey']:
            bot_answer = "Hi there! How can I help you today?"
            reflection = "⚠️ Possibly Irrelevant"
            log_step("plan", f"Greeting detected")
            log_step("reflect", reflection)
        else:
            chunks = retrieve_relevant_chunks(user_query)
            bot_answer = answer_node(user_query, chunks)
            reflection = reflect_node(user_query, bot_answer)
            evaluate_answer(user_query, bot_answer)

        print(f"\nBot: {bot_answer}\nReflection: {reflection}")

# ------------------- Entry point -------------------
if __name__ == "__main__":
    main()
