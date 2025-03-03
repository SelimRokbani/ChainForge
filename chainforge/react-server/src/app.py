import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import spacy
import io
import fitz
from docx import Document
import cohere
import gensim
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bertopic import BERTopic
from dotenv import load_dotenv
import logging
import math
import tempfile
import openai

# For Retrieval Methods
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

# Add these imports at the top
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt")
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model 'en_core_web_sm'.")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    logger.info("Downloaded and loaded spaCy model 'en_core_web_sm'.")

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
if COHERE_API_KEY:
    co = cohere.Client(COHERE_API_KEY)
    logger.info("Initialized Cohere client.")
else:
    co = None
    logger.warning("Cohere API key not found. Cohere-based chunking methods unavailable.")

openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    logger.warning("OpenAI API key not found. OpenAI-based embeddings will fail unless you set OPENAI_API_KEY.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------- File Extraction Helpers --------------------
def extract_text_from_pdf(file_bytes):
    try:
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def extract_text_from_docx(file_bytes):
    try:
        file_stream = io.BytesIO(file_bytes)
        doc = Document(file_stream)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {e}")
        raise

# -------------------- Chunking Methods --------------------
from transformers import AutoTokenizer
huggingface_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def overlapping_langchain_textsplitter(text, chunk_size=200, chunk_overlap=50, keep_separator=True):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=keep_separator
    )
    chunks = splitter.split_text(text)
    return chunks if chunks else [text]

def overlapping_openai_tiktoken(text, chunk_size=200, chunk_overlap=50):
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    result = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = enc.decode(tokens[start:end])
        result.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if end >= len(tokens):
            break
    return result if result else [text]

def overlapping_huggingface_tokenizers(text, chunk_size=200, chunk_overlap=50):
    tokens = huggingface_tokenizer.encode(text)
    result = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = huggingface_tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        result.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if end >= len(tokens):
            break
    return result if result else [text]

def syntax_spacy(text):
    doc = nlp.pipe([text])
    sents = []
    for d in doc:
        for s in d.sents:
            sent_text = s.text.strip()
            if sent_text:
                sents.append(sent_text)
    return sents if sents else [text]

def syntax_texttiling(text):
    from nltk.tokenize import TextTilingTokenizer
    tt = TextTilingTokenizer()
    chunks = tt.tokenize(text)
    return chunks if chunks else [text]

def hybrid_texttiling_spacy(text, chunk_size=300, chunk_overlap=100):
    from nltk.tokenize import TextTilingTokenizer
    tt = TextTilingTokenizer()
    sem_chunks = tt.tokenize(text) or [text]
    final_chunks = []
    for chunk in sem_chunks:
        doc = nlp(chunk)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        buf = []
        current_size = 0
        for sentence in sentences:
            if current_size + len(sentence) > chunk_size:
                final_chunks.append(" ".join(buf))
                buf = buf[-chunk_overlap:] + [sentence]
                current_size = sum(len(x) for x in buf)
            else:
                buf.append(sentence)
                current_size += len(sentence)
        if buf:
            final_chunks.append(" ".join(buf))
    return final_chunks if final_chunks else [text]

def semantic_bertopic(text, n_topics=2, min_topic_size=1):
    topic_model = BERTopic(n_topics=n_topics, min_topic_size=min_topic_size, random_state=42)
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [text]
    topics = topic_model.fit_transform(sentences)
    clusters = {}
    for t, s in zip(topics, sentences):
        clusters.setdefault(t, []).append(s)
    return [" ".join(v) for v in clusters.values()] or [text]

def hybrid_bertopic_spacy(text, min_topic_size=2):
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if not sentences:
        return [text]
    topic_model = BERTopic(min_topic_size=min_topic_size, random_state=42)
    topics = topic_model.fit_transform(sentences)
    clusters = {}
    for t, s in zip(topics, sentences):
        clusters.setdefault(t, []).append(s)
    return [" ".join(v) for v in clusters.values()] or [text]

def hybrid_recursive_gensim(text, chunk_size=300, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sem_chunks = splitter.split_text(text) or [text]
    final_chunks = []
    for sc in sem_chunks:
        sents = nltk.sent_tokenize(sc)
        processed = [simple_preprocess(s) for s in sents]
        dictionary = Dictionary(processed)
        if len(dictionary) == 0:
            continue
        corpus = [dictionary.doc2bow(d) for d in processed]
        from gensim.models import LdaModel
        lda = LdaModel(corpus, id2word=dictionary, num_topics=2, passes=10, random_state=42)
        assignments = []
        for bow in corpus:
            topic_probs = lda.get_document_topics(bow)
            if topic_probs:
                dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
            else:
                dominant_topic = 0
            assignments.append(dominant_topic)
        clusters = {}
        for lbl, st in zip(assignments, sents):
            clusters.setdefault(lbl, []).append(st)
        final_chunks.extend([" ".join(val) for val in clusters.values()])
    return final_chunks if final_chunks else [text]

def hybrid_recursive_cohere(text, chunk_size=300, chunk_overlap=100):
    if not co:
        return [text]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sem_chunks = splitter.split_text(text) or [text]
    final_chunks = []
    for sc in sem_chunks:
        sents = nltk.sent_tokenize(sc)
        if not sents:
            continue
        embeddings = co.embed(texts=sents).embeddings
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clusters = {}
        for lbl, st in zip(labels, sents):
            clusters.setdefault(lbl, []).append(st)
        final_chunks.extend([" ".join(val) for val in clusters.values()])
    return final_chunks if final_chunks else [text]

def hybrid_recursive_bertopic(text, chunk_size=300, chunk_overlap=300, n_topics=2, min_topic_size=1):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sem_chunks = splitter.split_text(text) or [text]
    final_chunks = []
    for sc in sem_chunks:
        sents = nltk.sent_tokenize(sc)
        if not sents:
            continue
        topic_model = BERTopic(n_topics=n_topics, min_topic_size=min_topic_size, random_state=42)
        topics, probs = topic_model.fit_transform(sents)
        clusters = {}
        for t, s0 in zip(topics, sents):
            clusters.setdefault(t, []).append(s0)
        final_chunks.extend([" ".join(val) for val in clusters.values()])
    return final_chunks if final_chunks else [text]

CHUNKING_METHODS = {
    ("Overlapping Chunking", "LangChain's TextSplitter"): overlapping_langchain_textsplitter,
    ("Overlapping Chunking", "OpenAI tiktoken"): overlapping_openai_tiktoken,
    ("Overlapping Chunking", "HuggingFace Tokenizers"): overlapping_huggingface_tokenizers,
    ("Syntax-Based Chunking", "spaCy Sentence Splitter"): syntax_spacy,
    ("Syntax-Based Chunking", "TextTilingTokenizer"): syntax_texttiling,
    ("Hybrid Chunking", "TextTiling + spaCy"): hybrid_texttiling_spacy,
    ("Hybrid Chunking", "BERTopic + spaCy"): hybrid_bertopic_spacy,
    ("Hybrid Chunking", "Recursive TextSplitter + Gensim"): hybrid_recursive_gensim,
    ("Hybrid Chunking", "Recursive TextSplitter + Cohere"): hybrid_recursive_cohere,
    ("Hybrid Chunking", "Recursive TextSplitter + BERTopic"): hybrid_recursive_bertopic,
}

def chunk_text(method_name, library, text, **kwargs):
    func = CHUNKING_METHODS.get((method_name, library))
    if not func:
        logger.warning(f"No chunking function for method='{method_name}', library='{library}'. Returning full text.")
        return [text]
    try:
        chunks = func(text, **kwargs)
        return chunks
    except TypeError:
        # Fallback if the function doesn't accept all kwargs
        return func(text)

# -------------------- Vector & Similarity Helpers --------------------
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    normA = math.sqrt(sum(x * x for x in a))
    normB = math.sqrt(sum(y * y for y in b))
    return dot / (normA * normB + 1e-10)    


def faiss_retrieval(embeddings, store_mode, store_path, query, top_k=5):
    try:# Try to load the index
        if store_mode == "load":
            if not os.path.exists(store_path):
                return jsonify({
                    "error": f"FAISS index not found at {store_path}",
                    "vectorstore_status": "error"
                }), 404
            
            try:
                vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
                
                # Verify embedding dimensions
                test_embedding = embeddings.embed_query("test")
                if vector_store.index.d != len(test_embedding):
                    return jsonify({
                        "error": "Embedding dimensions mismatch. Index was created with different embeddings.",
                        "vectorstore_status": "error"
                    }), 400
                
                # Perform similarity search
                results = vector_store.similarity_search_with_score(query, k=top_k)
                
                # Format results
                retrieved = [{
                    "text": doc.page_content,
                    "similarity": float(score),  # Convert numpy float to native float
                    "docTitle": doc.metadata.get("docTitle", ""),
                    "chunkId": doc.metadata.get("chunkId", str(i))
                } for i, (doc, score) in enumerate(results)]
                
                return jsonify({
                    "retrieved": retrieved,
                    "vectorstore_status": "connected"
                }), 200

            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
                return jsonify({
                    "error": f"Failed to load FAISS index: {str(e)}",
                    "vectorstore_status": "error"
                }), 500

        elif store_mode == "create":
            if not os.path.isdir(store_path):
                return jsonify({
                    "error": f"Directory not found: {store_path}",
                    "vectorstore_status": "error"
                }), 404

            try:
                # Create documents with metadata
                documents = [
                    Document(
                        page_content=text,
                        metadata={
                            "docTitle": obj.get("docTitle", ""),
                            "chunkId": obj.get("chunkId", str(i))
                        }
                    ) for i, (text, obj) in enumerate(zip(chunk_texts, chunk_objs))
                ]
                
                # Create and save the index
                vector_store = FAISS.from_documents(documents, embeddings)
                vector_store.save_local(store_path)
                
                # Perform similarity search
                results = vector_store.similarity_search_with_score(query, k=top_k)
                
                # Format results
                retrieved = [{
                    "text": doc.page_content,
                    "similarity": float(score),
                    "docTitle": doc.metadata.get("docTitle", ""),
                    "chunkId": doc.metadata.get("chunkId", "")
                } for doc, score in results]
                
                return jsonify({
                    "retrieved": retrieved,
                    "vectorstore_status": "connected"
                }), 200

            except Exception as e:
                logger.error(f"Error creating FAISS index: {str(e)}")
                return jsonify({
                    "error": f"Failed to create FAISS index: {str(e)}",
                    "vectorstore_status": "error"
                }), 500

    except Exception as e:
        logger.error(f"FAISS operation failed: {str(e)}")
        return jsonify({
            "error": f"FAISS operation failed: {str(e)}",
            "vectorstore_status": "error"
        }), 500

# -------------- Whoosh (Optional) --------------
whoosh_index_dir = tempfile.mkdtemp()
schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
if not os.path.exists(whoosh_index_dir):
    os.mkdir(whoosh_index_dir)
ix = index.create_in(whoosh_index_dir, schema)

# -------------- Flask Routes --------------

@app.route("/upload", methods=["POST"])
def upload_files():
    """Extract text from the uploaded file based on its extension."""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename.lower()
    ext = filename.rsplit(".", 1)[-1]
    file_bytes = file.read()
    if ext == "pdf":
        text = extract_text_from_pdf(file_bytes)
    elif ext == "txt":
        text = extract_text_from_txt(file_bytes)
    elif ext == "docx":
        text = extract_text_from_docx(file_bytes)
    else:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    return jsonify({"text": text}), 200

@app.route("/process", methods=["POST"])
def process_file():
    """
    Extract and chunk text using specified chunking method (methodName + library).
    Either a file or raw text can be provided.
    """
    file = request.files.get("file")
    method_name = request.form.get("methodName")
    library = request.form.get("library")
    text_input = request.form.get("text", None)

    settings = {}
    for k in request.form:
        if k not in ["methodName", "library", "text", "file"]:
            val = request.form.get(k)
            if k in ["chunk_size", "chunk_overlap", "n_topics", "min_topic_size"]:
                try:
                    val = int(val)
                except ValueError:
                    val = None
            elif k in ["keep_separator"]:
                val = (val.lower() == "true")
            settings[k] = val

    if not method_name or not library:
        return jsonify({"error": "Missing methodName or library"}), 400

    # Extract text
    if file:
        filename = file.filename.lower()
        ext = filename.rsplit(".", 1)[-1]
        file_bytes = file.read()
        if ext == "pdf":
            text = extract_text_from_pdf(file_bytes)
        elif ext == "txt":
            text = extract_text_from_txt(file_bytes)
        elif ext == "docx":
            text = extract_text_from_docx(file_bytes)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    elif text_input:
        text = text_input
    else:
        return jsonify({"error": "No file or text provided"}), 400

    # Chunk
    chunks = chunk_text(method_name, library, text, **{k: v for k, v in settings.items() if v is not None})
    return jsonify({
        "library": library,
        "settings": settings,
        "chunks": chunks
    }), 200

class EmbeddingModelRegistry:
    _models = {}
    
    @classmethod
    def register(cls, model_name):
        def decorator(embedding_func):
            cls._models[model_name] = embedding_func
            return embedding_func
        return decorator
        
    @classmethod
    def get_embedder(cls, model_name):
        return cls._models.get(model_name)
    
    @classmethod
    def list_models(cls):
        return list(cls._models.keys())

@EmbeddingModelRegistry.register("huggingface")
def huggingface_embedder(texts, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generate embeddings using HuggingFace Transformers.
    
    Args:
        texts: List of text strings to embed
        model_name: HuggingFace model name/path (default: sentence-transformers/all-mpnet-base-v2)
        
    Returns:
        List of embeddings for each text
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        logger.info(f"Using HuggingFace model: {model_name} for {len(texts)} texts")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for t in batch_texts:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, 
                                  max_length=512)  # Add max_length for safety
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use mean pooling by default
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                batch_embeddings.append(emb)
                
            embeddings.extend(batch_embeddings)
            
        return embeddings
    except Exception as e:
        logger.error(f"HuggingFace embedder failed: {str(e)}")
        raise ValueError(f"Failed to generate HuggingFace embeddings: {str(e)}")

@EmbeddingModelRegistry.register("OpenAI Embeddings")
def openai_embedder(texts, model_name="text-embedding-ada-002"):
    """
    Generate embeddings using OpenAI Embeddings.
    
    Args:
        texts: List of text strings to embed
        model_name: OpenAI embedding model to use (default: text-embedding-ada-002)
        
    Returns:
        List of embeddings for each text
    """
    try:
        import openai
        logger.info(f"Using OpenAI model: {model_name} for {len(texts)} texts")
        
        embeddings = []
        # Process in batches of 16 to stay within rate limits
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for t in batch_texts:
                resp = openai.Embedding.create(input=t, model=model_name)
                emb = resp["data"][0]["embedding"]
                batch_embeddings.append(emb)
                
            embeddings.extend(batch_embeddings)
            
        return embeddings
    except Exception as e:
        logger.error(f"OpenAI embedder failed: {str(e)}")
        raise ValueError(f"Failed to generate OpenAI embeddings: {str(e)}")

@EmbeddingModelRegistry.register("Cohere Embeddings")
def cohere_embedder(texts, model_name="embed-english-v2.0"):
    """
    Generate embeddings using Cohere Embeddings.
    
    Args:
        texts: List of text strings to embed
        model_name: Cohere embedding model to use (default: embed-english-v2.0)
        
    Returns:
        List of embeddings for each text
    """
    try:
        import cohere
        logger.info(f"Using Cohere model: {model_name} for {len(texts)} texts")
        
        # Get API key from environment or settings
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            from flask import current_app
            api_key = current_app.config.get("COHERE_API_KEY")
            
        if not api_key:
            raise ValueError("Cohere API key not found in environment or app config")
            
        co = cohere.Client(api_key)
        
        batch_size = 32  
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = co.embed(texts=batch_texts, model=model_name)
            embeddings.extend(response.embeddings)
            
        return embeddings
    except Exception as e:
        logger.error(f"Cohere embedder failed: {str(e)}")
        raise ValueError(f"Failed to generate Cohere embeddings: {str(e)}")

@EmbeddingModelRegistry.register("Sentence Transformers")
def sentence_transformers_embedder(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings using Sentence Transformers.
    
    Args:
        texts: List of text strings to embed
        model_name: Sentence Transformers model name (default: all-MiniLM-L6-v2)
        
    Returns:
        List of embeddings for each text
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Using SentenceTransformer model: {model_name} for {len(texts)} texts")
        
        model = SentenceTransformer(model_name)
        
        # Process in reasonable batch sizes
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = model.encode(batch_texts).tolist()
            embeddings.extend(embeddings)
            
        return embeddings
    except Exception as e:
        logger.error(f"SentenceTransformer embedder failed: {str(e)}")
        raise ValueError(f"Failed to generate SentenceTransformer embeddings: {str(e)}")

# Define a registry for retrieval methods
class RetrievalMethodRegistry:
    _methods = {}
    
    @classmethod
    def register(cls, method_name):
        def decorator(handler_func):
            cls._methods[method_name] = handler_func
            return handler_func
        return decorator
        
    @classmethod
    def get_handler(cls, method_name):
        return cls._methods.get(method_name)

@RetrievalMethodRegistry.register("bm25")
def handle_bm25(chunk_objs, queries, settings):
    top_k = settings.get("top_k", 5)
    k1 = settings.get("bm25_k1", 1.5)
    b = settings.get("bm25_b", 0.75)
    
    # Extract text from chunk objects
    chunk_texts = [chunk.get("text", "") for chunk in chunk_objs]
    
    # Preprocess corpus once
    tokenized_corpus = [simple_preprocess(doc) for doc in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    
    results = {}
    for query in queries:
        tokenized_query = simple_preprocess(query)
        raw_scores = bm25.get_scores(tokenized_query)
        
        # Normalize scores
        max_score = max(raw_scores) if raw_scores.any() and max(raw_scores) > 0 else 1
        normalized_scores = [score / max_score for score in raw_scores]
        
        # Build result objects with all the necessary metadata
        retrieved = []
        scored_chunks = sorted(zip(chunk_objs, normalized_scores), key=lambda x: x[1], reverse=True)
        
        for chunk, similarity in scored_chunks[:top_k]:
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(similarity),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results[query] = retrieved
    
    return results

@RetrievalMethodRegistry.register("tfidf")
def handle_tfidf(chunk_objs, queries, settings):
    top_k = settings.get("top_k", 5)
    max_features = settings.get("max_features", 500)
    
    # Extract text from chunk objects
    chunk_texts = [chunk.get("text", "") for chunk in chunk_objs]
    
    # Create and fit vectorizer once for all queries
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    
    results = {}
    for query in queries:
        query_vec = vectorizer.transform([query])
        sims = (tfidf_matrix * query_vec.T).toarray().flatten()
        
        # Normalize scores
        max_sim = sims.max() if sims.size > 0 and sims.max() > 0 else 1
        normalized_sims = sims / max_sim
        
        # Build result objects
        retrieved = []
        ranked_idx = normalized_sims.argsort()[::-1][:top_k]
        
        for i in ranked_idx:
            chunk = chunk_objs[i]
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(normalized_sims[i]),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results[query] = retrieved
    
    return results

@RetrievalMethodRegistry.register("boolean")
def handle_boolean(chunk_objs, queries, settings):
    top_k = settings.get("top_k", 5)
    required_match_count = settings.get("required_match_count", 1)
    
    # Extract text from chunk objects
    chunk_texts = [chunk.get("text", "") for chunk in chunk_objs]
    
    results = {}
    for query in queries:
        q_tokens = set(simple_preprocess(query))
        if len(q_tokens) < required_match_count:
            # Not enough tokens in query to match the required count
            results[query] = []
            continue
            
        scored = []
        for i, c in enumerate(chunk_texts):
            c_tokens = set(simple_preprocess(c))
            matches = len(q_tokens.intersection(c_tokens))
            if matches >= required_match_count:
                score = matches / (len(c_tokens) + 1e-9)  # Normalize by document length
                scored.append((i, score))
                
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        retrieved = []
        if scored:
            max_score = scored[0][1]
            for i, score in scored[:top_k]:
                chunk = chunk_objs[i]
                normalized_score = score / max_score if max_score > 0 else 0
                retrieved.append({
                    "text": chunk.get("text", ""),
                    "similarity": float(normalized_score),
                    "docTitle": chunk.get("docTitle", ""),
                    "chunkId": chunk.get("chunkId", ""),
                })
        
        results[query] = retrieved
    
    return results

@RetrievalMethodRegistry.register("overlap")
def handle_keyword_overlap(chunk_objs, queries, settings):
    top_k = settings.get("top_k", 5)
    
    # Extract text from chunk objects
    chunk_texts = [chunk.get("text", "") for chunk in chunk_objs]
    
    results = {}
    for query in queries:
        q_tokens = set(simple_preprocess(query))
        scored = []
        
        for i, c in enumerate(chunk_texts):
            c_tokens = set(simple_preprocess(c))
            overlap = len(q_tokens.intersection(c_tokens))
            scored.append((i, overlap))
            
        # Sort by overlap count
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        retrieved = []
        if scored and scored[0][1] > 0:  # Ensure max score > 0
            max_score = scored[0][1]
            for i, score in scored[:top_k]:
                chunk = chunk_objs[i]
                normalized_score = score / max_score
                retrieved.append({
                    "text": chunk.get("text", ""),
                    "similarity": float(normalized_score),
                    "docTitle": chunk.get("docTitle", ""),
                    "chunkId": chunk.get("chunkId", ""),
                })
        else:
            # No overlaps found
            retrieved = []
        
        results[query] = retrieved
    
    return results

import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans
import math


# Helper functions for similarity calculations
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

def manhattan_distance(vec1, vec2):
    """Compute Manhattan distance between two vectors"""
    return sum(abs(a - b) for a, b in zip(vec1, vec2))

def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two vectors"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

@RetrievalMethodRegistry.register("cosine")
def handle_cosine_similarity(chunks, chunk_embeddings, queries, query_embeddings, settings):
    """
    Retrieve chunks using cosine similarity between embeddings.
    
    This implementation uses a min-heap to keep only the top-k results in memory.
    """
    top_k = settings.get("top_k", 5)
    results = {}
    
    for (query, query_emb) in zip(queries, query_embeddings):
        # Use a min heap to keep track of top k results
        min_heap = []
        
        # Calculate similarities and maintain heap of size top_k
        for i, (chunk, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
            sim = cosine_similarity(chunk_emb, query_emb)
            
            # If heap is not full, add the item
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (sim, i))
            # If similarity is higher than the smallest in heap, replace it
            elif sim > min_heap[0][0]:
                heapq.heappushpop(min_heap, (sim, i))
        
        # Convert heap to sorted results (highest similarity first)
        retrieved = []
        for sim, i in sorted(min_heap, reverse=True):
            chunk = chunks[i]
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(sim),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results[query] = retrieved
    
    return results

@RetrievalMethodRegistry.register("manhattan")
def handle_manhattan(chunk_objs, chunk_embeddings, queries, query_embeddings, settings):
    """
    Retrieve chunks using Manhattan distance between embeddings.
    """
    top_k = settings.get("top_k", 5)
    results = {}
    
    for query, query_emb in zip(queries, query_embeddings):
        # Use a min heap to keep track of top k results
        min_heap = []
        
        # Calculate similarities and maintain heap of size top_k
        for i, (chunk, chunk_emb) in enumerate(zip(chunk_objs, chunk_embeddings)):
            # Lower Manhattan distance = higher similarity
            distance = manhattan_distance(chunk_emb, query_emb)
            sim = 1.0 / (1.0 + distance)  # Transform to similarity score
            
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (sim, i))
            elif sim > min_heap[0][0]:
                heapq.heappushpop(min_heap, (sim, i))
        
        # Convert heap to sorted results
        retrieved = []
        for sim, i in sorted(min_heap, reverse=True):
            chunk = chunk_objs[i]
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(sim),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results[query] = retrieved
    
    return results

@RetrievalMethodRegistry.register("euclidean")
def handle_euclidean(chunk_objs, chunk_embeddings, queries, query_embeddings, settings):
    """
    Retrieve chunks using Euclidean distance between embeddings.
    """
    top_k = settings.get("top_k", 5)
    results = {}
    
    for query, query_emb in zip(queries, query_embeddings):
        min_heap = []
        
        for i, (chunk, chunk_emb) in enumerate(zip(chunk_objs, chunk_embeddings)):
            distance = euclidean_distance(chunk_emb, query_emb)
            sim = 1.0 / (1.0 + distance)  # Transform to similarity score
            
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (sim, i))
            elif sim > min_heap[0][0]:
                heapq.heappushpop(min_heap, (sim, i))
        
        # Convert heap to sorted results
        retrieved = []
        for sim, i in sorted(min_heap, reverse=True):
            chunk = chunk_objs[i]
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(sim),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results[query] = retrieved
    
    return results

@RetrievalMethodRegistry.register("clustered")
def handle_clustered(chunk_objs, chunk_embeddings, queries, query_embeddings, settings):
    """
    Retrieve chunks using a combination of query similarity and cluster similarity.
    """
    top_k = settings.get("top_k", 5)
    n_clusters = settings.get("n_clusters", 3)
    query_coeff = settings.get("query_coeff", 0.6)
    center_coeff = settings.get("center_coeff", 0.4)
    results = {}
    
    # Convert embeddings to numpy array for clustering
    X = np.array(chunk_embeddings)
    
    # Only perform clustering if we have enough samples
    if len(X) >= 2:
        n_clusters = min(n_clusters, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_
        
        for query, query_emb in zip(queries, query_embeddings):
            min_heap = []
            query_emb_np = np.array(query_emb).reshape(1, -1)
            
            for i, (chunk, chunk_emb) in enumerate(zip(chunk_objs, chunk_embeddings)):
                # Calculate similarity to query
                chunk_emb_np = np.array(chunk_emb).reshape(1, -1)
                query_sim = float(sklearn_cosine(chunk_emb_np, query_emb_np)[0][0])
                
                # Calculate similarity to cluster center
                center_sim = float(sklearn_cosine(
                    chunk_emb_np, 
                    cluster_centers[labels[i]].reshape(1, -1)
                )[0][0])
                
                # Combined similarity score (weighted)
                combined_sim = query_coeff * query_sim + center_coeff * center_sim
                
                if len(min_heap) < top_k:
                    heapq.heappush(min_heap, (combined_sim, i))
                elif combined_sim > min_heap[0][0]:
                    heapq.heappushpop(min_heap, (combined_sim, i))
            
            # Convert heap to sorted results
            retrieved = []
            for sim, i in sorted(min_heap, reverse=True):
                chunk = chunk_objs[i]
                retrieved.append({
                    "text": chunk.get("text", ""),
                    "similarity": float(sim),
                    "docTitle": chunk.get("docTitle", ""),
                    "chunkId": chunk.get("chunkId", ""),
                })
            
            results[query] = retrieved
    return results

@app.route("/retrieve", methods=["POST"])
def retrieve():
    """
    Process multiple retrieval methods against provided chunks and queries.
    
    Expected request format:
    {
        "methods": [
            {
                "id": "unique_method_id",
                "baseMethod": "bm25",
                "methodName": "BM25",
                "library": "BM25",
                "settings": { "top_k": 5, ... }
            },
            {
                "id": "unique_method_id",
                "baseMethod": "cosine",
                "methodName": "Cosine similarity",
                "library": "Cosine",
                "embeddingProvider": "HuggingFace Transformers",
                "settings": { "embeddingModel": "all-MiniLM-L6-v2", "top_k": 5, ... }
            }
            ...
        ],
        "chunks": [{"text": "...", "docTitle": "...", "chunkId": "..."}, ...],
        "queries": ["query1", "query2", ...] 
    }

    Expected output format:
    {
        "results": {
            "method_id": {
                "baseMethod": "bm25",
                "methodName": "BM25",
                "retrieved": {"query1": [chunk1, chunk2, ...], "query2": [...], ...},
                "status": "success"
            },
            ...
        }
    }
    
    Returns progressive results as each method completes.
    """
    data = request.json
    methods = data.get("methods", [])
    chunks = data.get("chunks", [])
    queries = data.get("queries", [])
    print(methods)
    # Validate inputs
    if not methods:
        return jsonify({"error": "No retrieval methods provided"}), 400
    if not chunks:
        return jsonify({"error": "No chunks provided"}), 400
    if not queries:
        return jsonify({"error": "No queries provided"}), 400
    

    # Group methods by embedding model to avoid redundant embedding computation
    embedding_methods = {}  # model -> list of methods requiring this model
    keyword_methods = []    # methods not requiring embeddings

    for method in methods:
        try:
            embedding_provider= method.get("embeddingProvider", None)
            if embedding_provider:
                # This is an embedding-based method
                embedding_model = method.get("settings")["embeddingModel"]
                full_embedder = f"{embedding_provider}#{embedding_model}"    
                if full_embedder not in embedding_methods:
                    embedding_methods[full_embedder] = []
                embedding_methods[full_embedder].append(method)
            else:
                # Non-embedding method
                keyword_methods.append(method)
        except Exception as e:
            results[method_id] = {
                "baseMethod": base_method,
                "methodName": method.get("methodName"),
                "error": str(e),
                "status": "error"
            }
            
    
    # Prepare the response object
    results = {}

    # Process keyword methods
    for method in keyword_methods:
        method_id = method.get("id")
        base_method = method.get("baseMethod")
        
        try:
            handler = RetrievalMethodRegistry.get_handler(base_method)
            if not handler:
                raise ValueError(f"Unknown method: {base_method}")
            retrieved = handler(chunks, queries, method.get("settings", {}))
                
            results[method_id] = {
                "baseMethod": base_method,
                "methodName": method.get("methodName"),
                "retrieved": retrieved,
                "status": "success"
            }
        except Exception as e:
            # Handle errors
            results[method_id] = {
                "baseMethod": base_method,
                "methodName": method.get("methodName"),
                "error": str(e),
                "status": "error"
            }

    # Process embedding-based methods grouped by model
    for embedder, methods in embedding_methods.items():
        try:
            provider, model_name = embedder.split("#", 1)
            embedder = EmbeddingModelRegistry.get_embedder(provider)
            if not embedder:
                        raise ValueError(f"Unknown embedding model: {model_name}")
            
            # Compute embeddings once for all methods using this model
            chunk_embeddings = embedder([c["text"] for c in chunks])
            query_embeddings = embedder(queries)
        except Exception as e:
                # Mark all methods using this model as failed
                for method in methods:
                    results[method.get("id")] = {
                        "baseMethod": method.get("baseMethod"),
                        "methodName": method.get("methodName"),
                        "error": f"Embedding model error: {str(e)}",
                        "status": "error"
                    }
                continue
        
        # Process each method with the same embeddings
        for method in methods:
            method_id = method.get("id")
            base_method = method.get("baseMethod")
            
            try:
                handler = RetrievalMethodRegistry.get_handler(base_method)
                if not handler:
                    raise ValueError(f"Unknown method: {base_method}")                    
                retrieved = handler(chunks, chunk_embeddings, queries, query_embeddings, method.get("settings", {}))
                    
                results[method_id] = {
                    "baseMethod": base_method,
                    "methodName": method.get("methodName"),
                    "retrieved": retrieved,
                    "status": "success",
                    "embeddingModel": model_name
                }
            except Exception as e:
                # Handle errors
                results[method_id] = {
                    "baseMethod": base_method,
                    "methodName": method.get("methodName"),
                    "error": str(e),
                    "status": "error",
                    "embeddingModel": model_name
                }
    
    return jsonify({"results": results}), 200

if __name__ == "__main__":
    app.run(debug=True)