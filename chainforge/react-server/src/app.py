import os
from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk
import spacy

# Ensure required NLTK data and spaCy model are downloaded
nltk.download('punkt')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

import cohere
import gensim
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bertopic import BERTopic

# Initialize Cohere client
COHERE_API_KEY = "z3lNe5PDVxKT9DUz0UCrp4FGzdFeWPe9Q4cC1zwu" 
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



# Semantic Chunking - Gensim Topic Modeling
def semantic_gensim_topic(text):
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return [text]
        processed = [simple_preprocess(sent) for sent in sentences]
        dictionary = Dictionary(processed)
        if len(dictionary) == 0:
            return [text]
        corpus = [dictionary.doc2bow(doc) for doc in processed]
        lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=5, passes=15, random_state=42)
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
        chunks = [" ".join(c) for c in clusters.values()]
        return chunks
    except Exception as e:
        print(f"Gensim Topic Modeling failed: {e}")
        return [text]

# Semantic Chunking - Cohere Embeddings
def semantic_cohere_embeddings(text):
    if not co:
        return [text]  # fallback if no API key
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return [text]
        embeddings = co.embed(texts=sentences).embeddings
        if not embeddings:
            return [text]
        kmeans = KMeans(n_clusters=5, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clusters = {}
        for lbl, s in zip(labels, sentences):
            clusters.setdefault(lbl, []).append(s)
        chunks = [" ".join(c) for c in clusters.values()]
        return chunks
    except Exception as e:
        print(f"Cohere Embedding failed: {e}")
        return [text]

# Semantic Chunking - BERTopic
def semantic_bertopic(text):
    try:
        topic_model = BERTopic(n_topics=5, random_state=42)
        topics, probs = topic_model.fit_transform(nltk.sent_tokenize(text))
        clusters = {}
        for topic, sentence in zip(topics, nltk.sent_tokenize(text)):
            if topic == -1:
                clusters.setdefault('outlier', []).append(sentence)
            else:
                clusters.setdefault(topic, []).append(sentence)
        chunks = [" ".join(c) for c in clusters.values()]
        return chunks
    except Exception as e:
        print(f"BERTopic failed: {e}")
        return [text]

# Overlapping Chunking - LangChain's TextSplitter
def overlapping_langchain_textsplitter(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_text(text)
        if not chunks:
            return [text]
        return chunks
    except Exception as e:
        print(f"LangChain's TextSplitter failed: {e}")
        return [text]

# Overlapping Chunking - tiktoken
def overlapping_openai_tiktoken(text):
    try:
        enc = tiktoken.get_encoding("r50k_base")
        tokens = enc.encode(text)
        chunk_size = 200
        overlap = 50
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk = enc.decode(tokens[start:end])
            chunks.append(chunk)
            start = end - overlap
            if start < 0:
                start = 0
            if end >= len(tokens):
                break
        return chunks
    except Exception as e:
        print(f"OpenAI tiktoken chunking failed: {e}")
        return [text]

# Overlapping Chunking - HuggingFace Tokenizers
def overlapping_huggingface_tokenizers(text):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer.encode(text)
        chunk_size = 200
        overlap = 50
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            chunks.append(chunk)
            start = end - overlap
            if start < 0:
                start = 0
            if end >= len(tokens):
                break
        return chunks
    except Exception as e:
        print(f"HuggingFace Tokenizers chunking failed: {e}")
        return [text]

# Syntax-Based Chunking - spaCy Sentence Splitter
def syntax_spacy(text):
    try:
        doc = nlp(text)
        chunks = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not chunks:
            return [text]
        return chunks
    except Exception as e:
        print(f"spaCy Sentence Splitter failed: {e}")
        return [text]

# Syntax-Based Chunking - NLTK Sentence Splitter
def syntax_nltk(text):
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return [text]
        return sentences
    except Exception as e:
        print(f"NLTK Sentence Splitter failed: {e}")
        return [text]

# Syntax-Based Chunking - TextTilingTokenizer
def syntax_texttiling(text):
    try:
        from nltk.tokenize import TextTilingTokenizer
        tt = TextTilingTokenizer()
        chunks = tt.tokenize(text)
        if not chunks:
            return [text]
        return chunks
    except Exception as e:
        print(f"TextTilingTokenizer Sentence Splitter failed: {e}")
        return [text]

# Hybrid Chunking - TextTiling + spaCy
def hybrid_texttiling_spacy(text):
    try:
        from nltk.tokenize import TextTilingTokenizer
        tt = TextTilingTokenizer()
        sem_chunks = tt.tokenize(text)
        final_chunks = []
        for sc in sem_chunks:
            doc = nlp(sc)
            for sent in doc.sents:
                st = sent.text.strip()
                if st:
                    final_chunks.append(st)
        if not final_chunks:
            return [text]
        return final_chunks
    except Exception as e:
        print(f"Hybrid TextTiling + spaCy failed: {e}")
        return [text]

# Hybrid Chunking - BERTopic + spaCy
def hybrid_bertopic_spacy(text):
    try:
        sem_chunks = semantic_bertopic(text)
        final_chunks = []
        for sc in sem_chunks:
            doc = nlp(sc)
            for sent in doc.sents:
                st = sent.text.strip()
                if st:
                    final_chunks.append(st)
        if not final_chunks:
            return [text]
        return final_chunks
    except Exception as e:
        print(f"Hybrid BERTopic + spaCy failed: {e}")
        return [text]

# Hybrid Chunking - Recursive TextSplitter + Gensim
def hybrid_recursive_gensim(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        sem_chunks = splitter.split_text(text)
        final_chunks = []
        for sc in sem_chunks:
            sentences = nltk.sent_tokenize(sc)
            processed = [simple_preprocess(sent) for sent in sentences]
            dictionary = Dictionary(processed)
            if len(dictionary) == 0:
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
            return [text]
        return final_chunks
    except Exception as e:
        print(f"Hybrid Recursive TextSplitter + Gensim failed: {e}")
        return [text]

# Hybrid Chunking - Recursive TextSplitter + Cohere
def hybrid_recursive_cohere(text):
    if not co:
        return [text]
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        sem_chunks = splitter.split_text(text)
        final_chunks = []
        for sc in sem_chunks:
            sentences = nltk.sent_tokenize(sc)
            embeddings = co.embed(texts=sentences).embeddings
            if not embeddings:
                continue
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            clusters = {}
            for lbl, s in zip(labels, sentences):
                clusters.setdefault(lbl, []).append(s)
            sem_chunks = [" ".join(c) for c in clusters.values()]
            final_chunks.extend(sem_chunks)
        if not final_chunks:
            return [text]
        return final_chunks
    except Exception as e:
        print(f"Hybrid Recursive TextSplitter + Cohere failed: {e}")
        return [text]

# Hybrid Chunking - Recursive TextSplitter + BERTopic
def hybrid_recursive_bertopic(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        sem_chunks = splitter.split_text(text)
        final_chunks = []
        for sc in sem_chunks:
            topic_model = BERTopic(n_topics=2, random_state=42)
            topics, probs = topic_model.fit_transform(nltk.sent_tokenize(sc))
            clusters = {}
            for topic, sentence in zip(topics, nltk.sent_tokenize(sc)):
                if topic == -1:
                    clusters.setdefault('outlier', []).append(sentence)
                else:
                    clusters.setdefault(topic, []).append(sentence)
            sem_chunks = [" ".join(c) for c in clusters.values()]
            final_chunks.extend(sem_chunks)
        if not final_chunks:
            return [text]
        return final_chunks
    except Exception as e:
        print(f"Hybrid Recursive TextSplitter + BERTopic failed: {e}")
        return [text]



def chunk_text(method_name, library, text):
    print(f"DEBUG: chunk_text called with method_name='{method_name}', library='{library}'")
   
    if method_name == "Overlapping Chunking":
        if library == "LangChain's TextSplitter":
            return overlapping_langchain_textsplitter(text)
        elif library == "OpenAI tiktoken":
            return overlapping_openai_tiktoken(text)
        elif library == "HuggingFace Tokenizers":
            return overlapping_huggingface_tokenizers(text)
        else:
            return [text]

    elif method_name == "Syntax-Based Chunking":
        if library == "spaCy Sentence Splitter":
            return syntax_spacy(text)
        elif library == "TextTilingTokenizer":
            return syntax_texttiling(text)
        else:
            return [text]

    elif method_name == "Hybrid Chunking":
        if library == "TextTiling + spaCy":
            return hybrid_texttiling_spacy(text)
        elif library == "BERTopic + spaCy":
            return hybrid_bertopic_spacy(text)
        else:
            return [text]

    else:
        # fallback
        return [text]

@app.route('/process', methods=['POST'])
def process_file():
    file = request.files.get('file')
    method_name = request.form.get('methodName')
    library = request.form.get('library')

    if not file or not method_name or not library:
        return jsonify({"error": "Missing file, methodName, or library"}), 400

    try:
        text = file.read().decode('utf-8')
    except Exception as e:
        print(f"File reading/decoding failed: {e}")
        return jsonify({"error": "Failed to read or decode the file"}), 400

    chunks = chunk_text(method_name, library, text)
    return jsonify({"chunks": chunks}), 200

if __name__ == '__main__':
    app.run(debug=True)
