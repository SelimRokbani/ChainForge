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

def syntax_texttiling(text, **kwargs):
    from nltk.tokenize import TextTilingTokenizer
    w = kwargs.get("w", 20)
    k = kwargs.get("k", 10)
    threshold = kwargs.get("threshold", 0.1)
    avg_chunk_size = kwargs.get("chunk_size", 300)
    if not text or not text.strip():
        return [text]
    try:
        tt = TextTilingTokenizer(w=w, k=k, threshold=threshold)
        chunks = tt.tokenize(text)
        if (len(chunks) <= 1) and (len(text) > avg_chunk_size):
            if "\n\n" in text:
                fallback_chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
                if len(fallback_chunks) > 1:
                    logger.info("Fallback: split text via paragraph boundaries.")
                    return fallback_chunks
            import nltk
            sentences = nltk.sent_tokenize(text)
            group_chunks = []
            curr = ""
            for s in sentences:
                if len(curr) + len(s) < avg_chunk_size:
                    curr += " " + s
                else:
                    group_chunks.append(curr.strip())
                    curr = s
            if curr:
                group_chunks.append(curr.strip())
            logger.info("Fallback: split text by grouping sentences.")
            return group_chunks if group_chunks else [text]
        return chunks if chunks else [text]
    except Exception as e:
        logger.error(f"TextTiling error: {e}")
        # Final fallback: group sentences
        import nltk
        sentences = nltk.sent_tokenize(text)
        group_chunks = []
        curr = ""
        for s in sentences:
            if len(curr) + len(s) < avg_chunk_size:
                curr += " " + s
            else:
                group_chunks.append(curr.strip())
                curr = s
        if curr:
            group_chunks.append(curr.strip())
        return group_chunks if group_chunks else [text]

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
    # Replace n_topics with nr_topics for compatibility with current BERTopic API
    topic_model = BERTopic(nr_topics=n_topics, min_topic_size=min_topic_size, random_state=42)
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [text]
    topics, _ = topic_model.fit_transform(sentences)
    clusters = {}
    for t, s in zip(topics, sentences):
        clusters.setdefault(t, []).append(s)
    return [" ".join(v) for v in clusters.values()] or [text]

def hybrid_bertopic_spacy(text, min_topic_size=2):
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if not sentences:
        return [text]
    try:
        topic_model = BERTopic(min_topic_size=min_topic_size, random_state=42)
        topics = topic_model.fit_transform(sentences)
        clusters = {}
        for t, s in zip(topics, sentences):
            clusters.setdefault(t, []).append(s)
        return [" ".join(v) for v in clusters.values()] or [text]
    except Exception as e:
        logger.error(f"BERTopic error in hybrid_bertopic_spacy: {e}")
        doc = nlp(text)
        fallback = [s.text.strip() for s in doc.sents if s.text.strip()]
        return fallback if fallback else [text]

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

def hybrid_recursive_cohere(text, chunk_size=300, chunk_overlap=100, max_tokens=512, threshold=0.75):
    if not co:
        logger.error("Cohere client not available.")
        return [text]
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sem_chunks = splitter.split_text(text) or [text]
    final_chunks = []
    import nltk
    for chunk in sem_chunks:
        sents = nltk.sent_tokenize(chunk)
        # If too few sentences, do not cluster.
        if len(sents) < 2:
            final_chunks.append(chunk.strip())
            continue
        try:
            # Attempt to obtain embeddings from Cohere.
            response = co.embed(texts=sents)
            embeddings = response.embeddings
        except Exception as e:
            logger.error(f"Cohere embed error: {e}")
            final_chunks.append(chunk.strip())
            continue
        from sklearn.cluster import KMeans
        try:
            n_clusters = 2 if len(sents) >= 2 else 1
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            clusters = {}
            for label, sent in zip(labels, sents):
                clusters.setdefault(label, []).append(sent)
            final_chunks.extend([" ".join(clusters[label]) for label in clusters])
        except Exception as e:
            logger.error(f"Clustering error in hybrid_recursive_cohere: {e}")
            final_chunks.append(chunk.strip())
    return final_chunks if final_chunks else [text]

def hybrid_recursive_bertopic(text, chunk_size=300, chunk_overlap=300, n_topics=2, min_topic_size=1):
    if not text or not text.strip():
        return [text]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sem_chunks = splitter.split_text(text) or [text]
    final_chunks = []
    import nltk
    for sc in sem_chunks:
        sents = nltk.sent_tokenize(sc)
        if not sents:
            continue
        try:
            topic_model = BERTopic(nr_topics=n_topics, min_topic_size=min_topic_size, random_state=42)
            topics, _ = topic_model.fit_transform(sents)
            clusters = {}
            for t, sent in zip(topics, sents):
                clusters.setdefault(t, []).append(sent)
            final_chunks.extend([" ".join(val) for val in clusters.values()])
        except Exception as e:
            logger.error(f"BERTopic error in hybrid_recursive_bertopic: {e}")
            final_chunks.append(sc.strip())
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

def vectorize_texts(library, texts):
    """
    Uniform embedding function that returns the same dimension embeddings
    for any library: Hugging Face, OpenAI, Cohere, or Sentence Transformers.
    """
    logger.info(f"vectorize_texts(library={library}, #texts={len(texts)})")
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

        elif library == "OpenAI Embeddings":
            embeddings = []
            for t in texts:
                resp = openai.Embedding.create(input=t, model="text-embedding-ada-002")
                emb = resp["data"][0]["embedding"]
                embeddings.append(emb)
            return embeddings

        elif library == "Cohere Embeddings":
            if not co:
                raise ValueError("Cohere client not initialized or missing API key.")
            response = co.embed(texts=texts)
            return response.embeddings

        elif library == "Sentence Transformers":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(texts).tolist()
            return emb

        else:
            raise ValueError(f"Unknown vector library: {library}")
    except Exception as e:
        logger.error(f"vectorize_texts failed: {e}")
        raise

# -------------- Non-Vector Retrieval --------------
def bm25_retrieval(chunks, query, top_k=5):
    tokenized_corpus = [simple_preprocess(doc) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = simple_preprocess(query)
    raw_scores = bm25.get_scores(tokenized_query)
    max_score = max(raw_scores) if raw_scores.any() and max(raw_scores) > 0 else 1
    normalized_scores = [score / max_score for score in raw_scores]
    
    ranked = sorted(zip(chunks, normalized_scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def tfidf_retrieval(chunks, query, top_k=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)
    query_vec = vectorizer.transform([query])
    import numpy as np
    sims = (tfidf_matrix * query_vec.T).toarray().flatten()
    
    max_sim = sims.max() if sims.size > 0 and sims.max() > 0 else 1
    normalized_sims = sims / max_sim
    
    ranked_idx = normalized_sims.argsort()[::-1]
    return [(chunks[i], float(normalized_sims[i])) for i in ranked_idx[:top_k]]


def boolean_retrieval(chunks, query, top_k=5):
    q_tokens = set(simple_preprocess(query))
    scored = []
    for c in chunks:
        c_tokens = set(simple_preprocess(c))
        if q_tokens.issubset(c_tokens):
            score = len(q_tokens) / (len(c_tokens) + 1e-9)
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if scored:
        max_score = scored[0][1]
        normalized = [(doc, score / max_score if max_score > 0 else 0) for doc, score in scored]
        return normalized[:top_k]
    else:
        return scored

def keyword_overlap_retrieval(chunks, query, top_k=5):
    q_tokens = set(simple_preprocess(query))
    scored = []
    for c in chunks:
        c_tokens = set(simple_preprocess(c))
        overlap = len(q_tokens.intersection(c_tokens))
        scored.append((c, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if scored:
        max_score = scored[0][1]
        normalized = [(doc, score / max_score if max_score > 0 else 0) for doc, score in scored]
        return normalized[:top_k]
    else:
        return scored

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

@app.route("/retrieve", methods=["POST"])
def retrieve():
    data = request.json
    library = data.get("library")
    chunk_data = data.get("chunks", [])
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    method_type = data.get("type", "vectorization")
    retrieval_method = data.get("method", "cosine")

    # Convert chunk data
    chunk_texts = []
    chunk_objs = []
    for c in chunk_data:
        if isinstance(c, dict):
            chunk_texts.append(c.get("text", ""))
            chunk_objs.append(c)
        else:
            chunk_texts.append(str(c))
            chunk_objs.append({"text": str(c)})

    if method_type == "vectorization":
        # 1) Vectorize chunks and query
        chunk_embeddings = vectorize_texts(library, chunk_texts)
        query_embedding = vectorize_texts(library, [query])[0]

        if retrieval_method == "cosine":
            # Plain cosine similarity
            scored = []
            for i, emb in enumerate(chunk_embeddings):
                sim = cosine_similarity(emb, query_embedding)
                scored.append({
                    "text": chunk_texts[i],
                    "similarity": float(sim),
                    "docTitle": chunk_objs[i].get("docTitle", ""),
                    "chunkId": chunk_objs[i].get("chunkId", ""),
                })
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            retrieved = scored[:top_k]

        elif retrieval_method == "sentenceEmbeddings":
            # Use Manhattan distance instead of cosine similarity
            def manhattan_distance(a, b):
                return sum(abs(x - y) for x, y in zip(a, b))
            scored = []
            for i, emb in enumerate(chunk_embeddings):
                dist = manhattan_distance(emb, query_embedding)
                # Convert distance to a similarity-like score
                sim = 1.0 / (1.0 + dist)
                scored.append({
                    "text": chunk_texts[i],
                    "similarity": float(sim),
                    "docTitle": chunk_objs[i].get("docTitle", ""),
                    "chunkId": chunk_objs[i].get("chunkId", ""),
                })
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            retrieved = scored[:top_k]

        elif retrieval_method == "customVector":
            import math
            def euclidean_distance(a, b):
                return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
            scored = []
            for i, emb in enumerate(chunk_embeddings):
                dist = euclidean_distance(emb, query_embedding)
                # Smaller distance => higher relevance, so invert
                sim_score = 1.0 / (1.0 + dist)
                scored.append({
                    "text": chunk_texts[i],
                    "similarity": float(sim_score),
                    "docTitle": chunk_objs[i].get("docTitle", ""),
                    "chunkId": chunk_objs[i].get("chunkId", ""),
                })
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            retrieved = scored[:top_k]

        elif retrieval_method == "clustered":
            import numpy as np
            X = np.array(chunk_embeddings)
            n_clusters = min(2, len(X))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            cluster_centers = kmeans.cluster_centers_

            new_scored = []
            for i, emb in enumerate(chunk_embeddings):
                sim_query = cosine_similarity(emb, query_embedding)
                sim_center = cosine_similarity(emb, cluster_centers[labels[i]])
                combined = 0.6 * sim_query + 0.4 * sim_center
                new_scored.append({
                    "text": chunk_texts[i],
                    "similarity": float(combined),
                    "docTitle": chunk_objs[i].get("docTitle", ""),
                    "chunkId": chunk_objs[i].get("chunkId", ""),
                })
            new_scored.sort(key=lambda x: x["similarity"], reverse=True)
            retrieved = new_scored[:top_k]

        else:
            # fallback
            scored = []
            for i, emb in enumerate(chunk_embeddings):
                sim = cosine_similarity(emb, query_embedding)
                scored.append({
                    "text": chunk_texts[i],
                    "similarity": float(sim),
                    "docTitle": chunk_objs[i].get("docTitle", ""),
                    "chunkId": chunk_objs[i].get("chunkId", ""),
                })
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            retrieved = scored[:top_k]

        return jsonify({"retrieved": retrieved}), 200

    else:
        # method_type == "retrieval" => Keyword-based
        if library == "BM25":
            ranked = bm25_retrieval(chunk_texts, query, top_k=top_k)
        elif library == "TF-IDF":
            ranked = tfidf_retrieval(chunk_texts, query, top_k=top_k)
        elif library == "Boolean Search":
            ranked = boolean_retrieval(chunk_texts, query, top_k=top_k)
        elif library == "KeywordOverlap":
            ranked = keyword_overlap_retrieval(chunk_texts, query, top_k=top_k)
        else:
            return jsonify({"error": f"Unknown keyword-based library: {library}"}), 400

        retrieved = []
        for (chunk_str, score) in ranked:
            idx = chunk_texts.index(chunk_str)
            retrieved.append({
                "text": chunk_str,
                "similarity": score,
                "docTitle": chunk_objs[idx].get("docTitle", ""),
                "chunkId": chunk_objs[idx].get("chunkId", ""),
            })
        return jsonify({"retrieved": retrieved}), 200

if __name__ == "__main__":
    app.run(debug=True)