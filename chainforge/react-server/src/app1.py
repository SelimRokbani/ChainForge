from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from nltk.tokenize import sent_tokenize

from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import spacy

# Initialize Flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

@app.route('/chunk', methods=['POST'])
def chunk_text():
    try:
        # Retrieve file and parameters
        file = request.files.get('file')
        method = request.form.get('method')
        library = request.form.get('library')

        if not file or not method or not library:
            return jsonify({"error": "File, method, or library not provided"}), 400

        # Save file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Delete temporary file
        os.remove(file_path)

        # Debugging logs
        print(f"Method: {method}, Library: {library}")

        # Chunking logic
        chunks = []
        if method == "Semantic/Topic-Based Chunking":
            if library == "NLTK TextTiling":
                from nltk.tokenize import TextTilingTokenizer
                tokenizer = TextTilingTokenizer()
                chunks = tokenizer.tokenize(text)
            
        elif method == "Overlapping Chunking":
            if library == "LangChain's TextSplitter":
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = splitter.split_text(text)
        elif method == "Syntax-Based Chunking":
            if library == "NLTK":
                chunks = sent_tokenize(text)
            elif library == "spaCy":
                doc = nlp(text)
                chunks = [sent.text for sent in doc.sents]
        elif method == "Hybrid Chunking":
            if library == "LangChain’s Semantic Chunker":
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(text)
        else:
            return jsonify({"error": "Invalid method or library provided"}), 400

        return jsonify({"success": True, "chunks": chunks})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
