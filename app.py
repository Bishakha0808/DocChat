import os
import json
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from werkzeug.utils import secure_filename
import pandas as pd

# Langchain imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

OLLAMA_THREADS = min(os.cpu_count() or 4, 8)  # threads for LLM inference
EMBED_WORKERS  = min(os.cpu_count() or 4, 6)  # parallel embedding requests


class ParallelOllamaEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings with parallel embed_documents for faster ingestion."""
    def embed_documents(self, texts):
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
            return list(pool.map(self.embed_query, texts))

app = Flask(__name__, template_folder='uploads/templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

OLLAMA_BASE_URL = "http://localhost:11434"

# Global variables
vectorstore = None
chat_history = []  # list of {"role": "user"/"assistant", "content": "..."}

def process_document(file_path, filename):
    """Extracts text from PDF, DOCX, or XLSX and returns document chunks."""
    ext = filename.rsplit('.', 1)[1].lower()
    docs = []

    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext in ['doc', 'docx']:
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
        text = df.to_string()
        docs = [Document(page_content=text, metadata={"source": filename})]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,   # larger chunks = fewer total chunks to embed
        chunk_overlap=100, # reduced overlap
    )
    splits = text_splitter.split_documents(docs)
    return splits


def get_context_from_vectorstore(query, k=3):
    """Retrieve relevant chunks from vectorstore."""
    if vectorstore is None:
        return ""
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc in results])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def list_models():
    """Return list of available Ollama models."""
    try:
        resp = http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m['name'] for m in resp.json().get('models', [])
                  if 'embed' not in m['name']]
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'models': ['llama3', 'gemma'], 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    global vectorstore, chat_history

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        splits = process_document(file_path, filename)
        # ParallelOllamaEmbeddings embeds all chunks concurrently — much faster
        embeddings = ParallelOllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        chat_history = []  # reset history on new document
        return jsonify({'message': f'{filename} processed — {len(splits)} chunks indexed.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint using SSE."""
    global chat_history

    data = request.json
    query = data.get('query', '').strip()
    model_name = data.get('model', 'llama3')

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    if vectorstore is None:
        return jsonify({'error': 'Please upload a document first.'}), 400

    context = get_context_from_vectorstore(query, k=3)
    system_prompt = (
        "You are a concise, helpful assistant. Answer based ONLY on the document "
        "context below. Be direct and avoid unnecessary repetition.\n\n"
        f"Context:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-4:]:  # last 2 exchanges keeps context short
        messages.append(turn)
    messages.append({"role": "user", "content": query})

    # Ollama inference options for speed
    ollama_options = {
        "num_thread": OLLAMA_THREADS,
        "num_ctx":    2048,   # context window — enough for 3 chunks + history
        "temperature": 0.2,
    }

    def generate():
        full_response = ""
        try:
            resp = http_requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model_name, "messages": messages,
                      "stream": True, "options": ollama_options},
                stream=True,
                timeout=300
            )
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    token = chunk.get('message', {}).get('content', '')
                    if token:
                        full_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if chunk.get('done'):
                        break
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # Save to history after complete response
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": full_response})
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/chat/clear', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({'message': 'History cleared.'})

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)