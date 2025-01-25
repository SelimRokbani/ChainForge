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

# For Retrieval Methods
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model 'en_core_web_sm'.")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    logger.info("Downloaded and loaded spaCy model 'en_core_web_sm'.")

# Initialize Cohere client securely
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
if COHERE_API_KEY:
    co = cohere.Client(COHERE_API_KEY)
    logger.info("Initialized Cohere client.")
else:
    co = None
    logger.warning(
        "Cohere API key not found. Cohere-based chunking methods will be unavailable."
    )

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------- Text Extraction --------------------

# A helper function to extract text from PDF files
def extract_text_from_pdf(file_bytes):
    try:
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        logger.info("Extracted text from PDF.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

# A helper function to extract text from DOCX files
def extract_text_from_docx(file_bytes):
    try:
        file_stream = io.BytesIO(file_bytes)
        doc = Document(file_stream)
        full_text = [para.text for para in doc.paragraphs]
        text = "\n".join(full_text)
        logger.info("Extracted text from DOCX.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise

# A helper function to extract text from TXT files
def extract_text_from_txt(file_bytes):
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        logger.info("Extracted text from TXT.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {e}")
        raise

# -------------------- Chunking Methods --------------------

def overlapping_langchain_textsplitter(text, chunk_size=200, chunk_overlap=50, keep_separator=True):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=keep_separator
        )
        chunks = splitter.split_text(text)
        if not chunks:
            logger.warning("LangChain's TextSplitter produced no chunks.")
            return [text]
        logger.info(f"LangChain's TextSplitter produced {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"LangChain's TextSplitter failed: {e}")
        return [text]

def overlapping_openai_tiktoken(text, chunk_size=200, chunk_overlap=50):
    try:
        enc = tiktoken.get_encoding("r50k_base")
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk = enc.decode(tokens[start:end])
            chunks.append(chunk)
            start = end - chunk_overlap
            if start < 0:
                start = 0
            if end >= len(tokens):
                break
        if not chunks:
            logger.warning("OpenAI tiktoken produced no chunks.")
            return [text]
        logger.info(f"OpenAI tiktoken produced {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"OpenAI tiktoken chunking failed: {e}")
        return [text]

def overlapping_huggingface_tokenizers(text, chunk_size=200, chunk_overlap=50):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            chunks.append(chunk)
            start = end - chunk_overlap
            if start < 0:
                start = 0
            if end >= len(tokens):
                break
        if not chunks:
            logger.warning("HuggingFace Tokenizers produced no chunks.")
            return [text]
        logger.info(f"HuggingFace Tokenizers produced {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"HuggingFace Tokenizers chunking failed: {e}")
        return [text]

def syntax_spacy(text):
    try:
        doc = nlp(text)
        chunks = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not chunks:
            logger.warning("spaCy Sentence Splitter produced no chunks.")
            return [text]
        logger.info(f"spaCy Sentence Splitter produced {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"spaCy Sentence Splitter failed: {e}")
        return [text]

def syntax_texttiling(text):
    try:
        from nltk.tokenize import TextTilingTokenizer
        tt = TextTilingTokenizer()
        chunks = tt.tokenize(text)
        if not chunks:
            logger.warning("TextTilingTokenizer Sentence Splitter produced no chunks.")
            return [text]
        logger.info(f"TextTilingTokenizer Sentence Splitter produced {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"TextTilingTokenizer Sentence Splitter failed: {e}")
        return [text]

def hybrid_texttiling_spacy(text, chunk_size=300, chunk_overlap=100):
    try:
        from nltk.tokenize import TextTilingTokenizer
        tt = TextTilingTokenizer()
        sem_chunks = tt.tokenize(text)
        logger.info(f"TextTiling split text into {len(sem_chunks)} semantic chunks.")
        final_chunks = sem_chunks
        if not final_chunks:
            logger.warning("Hybrid TextTiling + spaCy produced no chunks.")
            return [text]
        logger.info(f"Hybrid TextTiling + spaCy produced {len(final_chunks)} final chunks.")
        return final_chunks
    except Exception as e:
        logger.error(f"Hybrid TextTiling + spaCy failed: {e}")
        return [text]

def semantic_bertopic(text, n_topics=2, min_topic_size=1):
    try:
        topic_model = BERTopic(n_topics=n_topics, min_topic_size=min_topic_size, random_state=42)
        topics, probs = topic_model.fit_transform(nltk.sent_tokenize(text))
        clusters = {}
        for topic, sentence in zip(topics, nltk.sent_tokenize(text)):
            if topic == -1:
                clusters.setdefault("outlier", []).append(sentence)
            else:
                clusters.setdefault(topic, []).append(sentence)
        sem_chunks = [" ".join(c) for c in clusters.values()]
        if not sem_chunks:
            logger.warning("BERTopic produced no semantic chunks.")
            return [text]
        logger.info(f"BERTopic produced {len(sem_chunks)} semantic chunks.")
        topic_counts = {topic: len(sents) for topic, sents in clusters.items()}
        logger.info(f"BERTopic topic distribution: {topic_counts}")
        return sem_chunks
    except Exception as e:
        logger.error(f"BERTopic failed: {e}")
        return [text]

def hybrid_bertopic_spacy(text, min_topic_size=2):
    try:
        sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        if not sentences:
            logger.warning("spaCy did not split the text into any sentences.")
            return [text]
        logger.info(f"spaCy produced {len(sentences)} sentences.")
        topic_model = BERTopic(min_topic_size=min_topic_size, random_state=42)
        topics, probs = topic_model.fit_transform(sentences)
        logger.info(f"BERTopic assigned topics: {topics}")
        clusters = {}
        for topic, sentence in zip(topics, sentences):
            if topic == -1:
                clusters.setdefault("outlier", []).append(sentence)
            else:
                clusters.setdefault(topic, []).append(sentence)
        sem_chunks = [" ".join(cluster) for cluster in clusters.values()]
        if not sem_chunks:
            logger.warning("Hybrid BERTopic + spaCy produced no chunks.")
            return [text]
        logger.info(f"Hybrid BERTopic + spaCy produced {len(sem_chunks)} chunks.")
        return sem_chunks
    except Exception as e:
        logger.error(f"Hybrid BERTopic + spaCy failed: {e}")
        return [text]

def hybrid_recursive_gensim(text, chunk_size=300, chunk_overlap=100):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        sem_chunks = splitter.split_text(text)
        logger.info(f"Recursive TextSplitter split text into {len(sem_chunks)} semantic chunks.")
        final_chunks = []
        for sc in sem_chunks:
            sentences = nltk.sent_tokenize(sc)
            processed = [simple_preprocess(sent) for sent in sentences]
            dictionary = Dictionary(processed)
            if len(dictionary) == 0:
                logger.warning("No tokens found after preprocessing in Gensim for a chunk.")
                continue
            corpus = [dictionary.doc2bow(doc) for doc in processed]
            lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=2, passes=10, random_state=42)
            topic_assignments = []
            for bow in corpus:
                topic_probs = lda.get_document_topics(bow)
                if topic_probs:
                    dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
                else:
                    dominant_topic = 0
                topic_assignments.append(dominant_topic)
            clusters = {}
            for lbl, s in zip(topic_assignments, sentences):
                clusters.setdefault(lbl, []).append(s)
            sem_chunks = [" ".join(c) for c in clusters.values()]
            final_chunks.extend(sem_chunks)
        if not final_chunks:
            logger.warning("Hybrid Recursive TextSplitter + Gensim produced no chunks.")
            return [text]
        logger.info(f"Hybrid Recursive TextSplitter + Gensim produced {len(final_chunks)} final chunks.")
        return final_chunks
    except Exception as e:
        logger.error(f"Hybrid Recursive TextSplitter + Gensim failed: {e}")
        return [text]

def hybrid_recursive_cohere(text, chunk_size=300, chunk_overlap=100):
    if not co:
        logger.error("Cohere client not initialized for hybrid_recursive_cohere.")
        return [text]
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        sem_chunks = splitter.split_text(text)
        logger.info(f"Recursive TextSplitter split text into {len(sem_chunks)} semantic chunks.")
        final_chunks = []
        for sc in sem_chunks:
            sentences = nltk.sent_tokenize(sc)
            embeddings = co.embed(texts=sentences).embeddings
            if not embeddings:
                logger.warning("No embeddings received from Cohere for a chunk.")
                continue
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            clusters = {}
            for lbl, s in zip(labels, sentences):
                clusters.setdefault(lbl, []).append(s)
            sem_chunks = [" ".join(c) for c in clusters.values()]
            final_chunks.extend(sem_chunks)
        if not final_chunks:
            logger.warning("Hybrid Recursive TextSplitter + Cohere produced no chunks.")
            return [text]
        logger.info(f"Hybrid Recursive TextSplitter + Cohere produced {len(final_chunks)} final chunks.")
        return final_chunks
    except Exception as e:
        logger.error(f"Hybrid Recursive TextSplitter + Cohere failed: {e}")
        return [text]

def hybrid_recursive_bertopic(text, chunk_size=300, chunk_overlap=300, n_topics=2, min_topic_size=1):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        sem_chunks = splitter.split_text(text)
        logger.info(f"Recursive TextSplitter split text into {len(sem_chunks)} semantic chunks.")
        final_chunks = []
        for sc in sem_chunks:
            topic_model = BERTopic(n_topics=n_topics, min_topic_size=min_topic_size, random_state=42)
            topics, probs = topic_model.fit_transform(nltk.sent_tokenize(sc))
            clusters = {}
            for topic, sentence in zip(topics, nltk.sent_tokenize(sc)):
                if topic == -1:
                    clusters.setdefault("outlier", []).append(sentence)
                else:
                    clusters.setdefault(topic, []).append(sentence)
            sem_chunks = [" ".join(c) for c in clusters.values()]
            final_chunks.extend(sem_chunks)
        if not final_chunks:
            logger.warning("Hybrid Recursive TextSplitter + BERTopic produced no chunks.")
            return [text]
        logger.info(f"Hybrid Recursive TextSplitter + BERTopic produced {len(final_chunks)} final chunks.")
        return final_chunks
    except Exception as e:
        logger.error(f"Hybrid Recursive TextSplitter + BERTopic failed: {e}")
        return [text]

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
    logger.info(f"Chunking called with method='{method_name}', library='{library}', kwargs={kwargs}")
    chunking_function = CHUNKING_METHODS.get((method_name, library))
    if not chunking_function:
        logger.warning(f"No chunking function found for method '{method_name}' & library '{library}'. Returning original text.")
        return [text]
    try:
        chunks = chunking_function(text, **kwargs)
        logger.info(f"Chunking function '{chunking_function.__name__}' produced {len(chunks)} chunks.")
        return chunks
    except TypeError as e:
        logger.error(f"TypeError in chunk_text: {e}. Attempting to run without kwargs.")
        try:
            chunks = chunking_function(text)
            logger.info(f"Chunking function '{chunking_function.__name__}' produced {len(chunks)} chunks without kwargs.")
            return chunks
        except Exception as ex:
            logger.error(f"Failed to run chunking function '{chunking_function.__name__}' without kwargs: {ex}")
            return [text]
    except Exception as e:
        logger.error(f"Error running chunking function '{chunking_function.__name__}': {e}")
        return [text]

# -------------------- Vectorization & Similarity Helpers --------------------

def vectorize_texts(library, texts):
    logger.info(f"vectorize_texts called with library='{library}' and {len(texts)} texts.")
    try:
        if library == "HuggingFace Transformers":
            from transformers import AutoTokenizer, AutoModel
            import torch
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            embeddings = []
            for t in texts:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                embeddings.append(emb)
            return embeddings
        elif library == "Sentence Transformers":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(texts).tolist()
            return emb
        elif library == "Custom KMeans":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            n_components = min(tfidf_matrix.shape[1], 300)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_matrix = svd.fit_transform(tfidf_matrix)
            logger.info(f"Reduced matrix shape after SVD: {reduced_matrix.shape}")
            return reduced_matrix.tolist()
        elif library == "OpenAI Embeddings":
            # Dummy example; insert proper OpenAI embedding call here.
            embeddings = [[0.1]*768 for _ in texts]
            return embeddings
        else:
            raise ValueError(f"Unknown library: {library}")
    except Exception as e:
        logger.error(f"Error in vectorize_texts: {e}")
        raise

def cosine_similarity(vecA, vecB):
    dot = sum(a * b for a, b in zip(vecA, vecB))
    normA = math.sqrt(sum(a * a for a in vecA))
    normB = math.sqrt(sum(b * b for b in vecB))
    return dot / (normA * normB + 1e-10)

# Initialize temporary directory for Whoosh index
whoosh_index_dir = tempfile.mkdtemp()
schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
if not os.path.exists(whoosh_index_dir):
    os.mkdir(whoosh_index_dir)
ix = index.create_in(whoosh_index_dir, schema)

def bm25_retrieval(chunks, query, top_k=5):
    tokenized_corpus = [simple_preprocess(doc) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = simple_preprocess(query)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    retrieved = [{"text": text, "similarity": score} for text, score in ranked[:top_k]]
    return retrieved

def tfidf_retrieval(chunks, query, top_k=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)
    query_vec = vectorizer.transform([query])
    cosine_similarities = (tfidf_matrix * query_vec.T).toarray().flatten()
    ranked_indices = cosine_similarities.argsort()[::-1]
    retrieved = [{"text": chunks[i], "similarity": float(cosine_similarities[i])} for i in ranked_indices[:top_k]]
    return retrieved

def boolean_retrieval(chunks, query, top_k=5):
    query_tokens = set(simple_preprocess(query))
    scored = []
    for c in chunks:
        tokens = set(simple_preprocess(c))
        if query_tokens.issubset(tokens):
            score = len(query_tokens) / len(tokens) if tokens else 0
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    retrieved = [{"text": text, "similarity": score} for text, score in scored[:top_k]]
    return retrieved

def keyword_overlap_retrieval(chunks, query, top_k=5):
    query_tokens = set(simple_preprocess(query))
    scored = []
    for c in chunks:
        tokens = set(simple_preprocess(c))
        overlap = len(query_tokens.intersection(tokens))
        scored.append((c, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    retrieved = [{"text": text, "similarity": overlap} for text, overlap in scored[:top_k]]
    return retrieved

# -------------------- Routes --------------------

@app.route("/vectorize", methods=["POST"])
def vectorize_chunks():
    # Receives chunks and a chosen library, returns vector embeddings
    data = request.json
    library = data.get("library")
    chunks = data.get("chunks")
    if not library or not chunks:
        return jsonify({"error": "Missing library or chunks"}), 400
    try:
        # Vectorize the chunks using the chosen library
        embeddings = vectorize_texts(library, chunks)
        return jsonify({"embeddings": embeddings}), 200
    except Exception as e:
        logger.error(f"Vectorization failed for library '{library}': {e}")
        return jsonify({"error": f"Vectorization failed: {str(e)}"}), 500

@app.route("/retrieve", methods=["POST"])
def retrieve():
    # Handles various retrieval methods (e.g., BM25, TF-IDF, vector-based)
    data = request.json
    library = data.get("library")
    chunks = data.get("chunks", [])
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    method_type = data.get("type", "vectorization")
    retrieval_method = data.get("method", "cosine")  
    if not library or not chunks or not query:
        return jsonify({"error": "Missing library, chunks, or query"}), 400

    try:
        if method_type == "vectorization":
            # Compute or generate embeddings, then rank by similarity
            if retrieval_method == "cosine":
                if library == "HuggingFace Transformers":
                    from transformers import AutoTokenizer, AutoModel
                    import torch
                    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                    chunk_embeddings = []
                    for t in chunks:
                        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
                        with torch.no_grad():
                            outputs = model(**inputs)
                        emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                        chunk_embeddings.append(emb)
                    query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        query_outputs = model(**query_inputs)
                    query_embedding = query_outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                else:
                    chunk_embeddings = vectorize_texts(library, chunks)
                    query_embedding = vectorize_texts(library, [query])[0]
            # 2. Sentence Embeddings: Use the SentenceTransformer model
            elif retrieval_method == "sentenceEmbeddings":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                chunk_embeddings = model.encode(chunks).tolist()
                query_embedding = model.encode([query])[0].tolist()
            # 3. Custom Vector: Use TF-IDF + SVD 
            elif retrieval_method == "customVector":
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import TruncatedSVD
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(chunks)
                n_components = min(tfidf_matrix.shape[1], 300)
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                chunk_embeddings = svd.fit_transform(tfidf_matrix).tolist()
                query_tfidf = vectorizer.transform([query])
                query_embedding = svd.transform(query_tfidf)[0].tolist()
            # 4. Clustered: Use default vectorization and then perform clustering 
            elif retrieval_method == "clustered":
                from transformers import AutoTokenizer, AutoModel
                import torch
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                chunk_embeddings = []
                for t in chunks:
                    inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                    chunk_embeddings.append(emb)
                query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    query_outputs = model(**query_inputs)
                query_embedding = query_outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

                # Perform clustering on chunk_embeddings
                from sklearn.cluster import KMeans
                import numpy as np
                X = np.array(chunk_embeddings)
                n_clusters = min(2, len(chunks))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(X)
                # For each cluster, compute an average embedding and re-score within the cluster.
                cluster_centers = kmeans.cluster_centers_
                new_scored = []
                for i, emb in enumerate(chunk_embeddings):
                    # Compute cosine similarity to query
                    sim_query = cosine_similarity(emb, query_embedding)
                    # Compute cosine similarity to the cluster center
                    sim_center = cosine_similarity(emb, cluster_centers[labels[i]])
                    # Combine the scores (tweak weights as needed)
                    sim = 0.6 * sim_query + 0.4 * sim_center
                    new_scored.append({"text": chunks[i], "similarity": float(sim)})
                new_scored.sort(key=lambda x: x["similarity"], reverse=True)
                return jsonify({"retrieved": new_scored[:top_k]}), 200
            else:
                from transformers import AutoTokenizer, AutoModel
                import torch
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                chunk_embeddings = []
                for t in chunks:
                    inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                    chunk_embeddings.append(emb)
                query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    query_outputs = model(**query_inputs)
                query_embedding = query_outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

            # Compute cosine similarity for the chosen branch
            scored = []
            for i, _ in enumerate(chunks):
                sim = cosine_similarity(chunk_embeddings[i], query_embedding)
                scored.append({"text": chunks[i], "similarity": float(sim)})
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            retrieved = scored[:top_k]

        elif method_type == "retrieval":
            # Non-vectorization retrieval strategies.
            if library in ["BM25", "TF-IDF"]:
                if library == "BM25":
                    retrieved = bm25_retrieval(chunks, query, top_k)
                else:
                    retrieved = tfidf_retrieval(chunks, query, top_k)
            elif library == "Boolean Search":
                retrieved = boolean_retrieval(chunks, query, top_k)
            elif library == "KeywordOverlap":
                retrieved = keyword_overlap_retrieval(chunks, query, top_k)
            else:
                chunk_embeddings = vectorize_texts(library, chunks)
                query_embedding = vectorize_texts(library, [query])[0]
                scored = []
                for i, _ in enumerate(chunks):
                    sim = cosine_similarity(chunk_embeddings[i], query_embedding)
                    scored.append({"text": chunks[i], "similarity": float(sim)})
                scored.sort(key=lambda x: x["similarity"], reverse=True)
                retrieved = scored[:top_k]
        else:
            raise ValueError(f"Unknown method type: {method_type}")

        return jsonify({"retrieved": retrieved}), 200
    except Exception as e:
        logger.error(f"Retrieval failed for library '{library}': {e}")
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload_files():
    # Extract text from uploaded file based on its extension
    file = request.files.get("file")
    if not file:
        logger.error("No file uploaded.")
        return jsonify({"error": "No file uploaded"}), 400
    filename = file.filename.lower()
    ext = filename.rsplit(".", 1)[-1]
    try:
        if ext == "pdf":
            file_bytes = file.read()
            text = extract_text_from_pdf(file_bytes)
        elif ext == "txt":
            file_bytes = file.read()
            text = extract_text_from_txt(file_bytes)
        elif ext == "docx":
            file_bytes = file.read()
            text = extract_text_from_docx(file_bytes)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return jsonify({"error": "Failed to extract text"}), 500
    logger.info(f"Extracted text with length {len(text)} characters.")
    return jsonify({"text": text}), 200

@app.route("/process", methods=["POST"])
def process_file():
    # Extract and chunk text using specified chunking method
    file = request.files.get("file")
    method_name = request.form.get("methodName")
    library = request.form.get("library")
    text_input = request.form.get("text", None)
    settings = {}
    for key in request.form:
        if key not in ["methodName", "library", "text"]:
            value = request.form.get(key)
            if key in ["chunk_size", "chunk_overlap", "n_topics", "min_topic_size", "depth", "window"]:
                try:
                    value = int(value)
                except ValueError:
                    logger.warning(f"Invalid integer for {key}: {value}. Using default.")
                    value = None
            elif key in ["keep_separator", "add_special_tokens", "padding"]:
                value = value.lower() == "true"
            settings[key] = value
    logger.info(f"Received request with methodName='{method_name}', library='{library}', settings={settings}")
    if not method_name or not library:
        logger.error("Missing methodName or library in the request.")
        return jsonify({"error": "Missing methodName or library"}), 400
    if file:
        try:
            filename = file.filename.lower()
            ext = filename.rsplit(".", 1)[-1]
            if ext == "pdf":
                file_bytes = file.read()
                text = extract_text_from_pdf(file_bytes)
            elif ext == "txt":
                file_bytes = file.read()
                text = extract_text_from_txt(file_bytes)
            elif ext == "docx":
                file_bytes = file.read()
                text = extract_text_from_docx(file_bytes)
            else:
                logger.error(f"Unsupported file type: {ext}")
                return jsonify({"error": f"Unsupported file type: {ext}"}), 400
        except Exception as e:
            logger.error(f"File reading/decoding failed: {e}")
            return jsonify({"error": "Failed to read or decode the file"}), 400
    elif text_input:
        text = text_input
        logger.info("Using text input from form.")
    else:
        logger.error("No file or text provided in the request.")
        return jsonify({"error": "No file or text provided"}), 400
    logger.info(f"Processing text with length {len(text)} characters.")
    chunks = chunk_text(method_name, library, text, **settings)
    logger.info(f"Generated {len(chunks)} chunks using method '{method_name}' and library '{library}'.")
    logger.debug(f"Chunks: {chunks}")
    return jsonify({
        "library": library,
        "settings": settings,
        "chunks": chunks
    }), 200

if __name__ == "__main__":
    # Run the Flask app for debugging
    app.run(debug=True)
