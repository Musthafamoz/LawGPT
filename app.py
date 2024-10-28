from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import logging
import os
import numpy as np
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up Flask and CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFQuestionAnswering:
    def __init__(self, model_path: str, pdf_path: str):
        """Initialize the PDF Question Answering system."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Download required NLTK data
        for resource in ['stopwords', 'wordnet', 'punkt']:
            nltk.download(resource, quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Initialize models
        self.llm = Llama(
            model_path=model_path,
            chat_format="chatml",
            n_ctx=2048
        )
        self.embedding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

        # Process PDF
        self.process_pdf(pdf_path)

    def process_pdf(self, pdf_path: str):
        """Process PDF and create embeddings."""
        reader = PdfReader(pdf_path)
        text_chunks = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks = self.chunk_text(text)
                text_chunks.extend(chunks)
        
        if not text_chunks:
            raise ValueError("No text content extracted from PDF")
        
        self.text_chunks = text_chunks
        self.chunk_embeddings = np.array(self.embedding_model.encode(text_chunks))

    def chunk_text(self, text: str, chunk_size: int = 400) -> list:
        """Split text into chunks."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = words
                current_length = len(words)
            else:
                current_chunk.extend(words)
                current_length += len(words)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def answer_question(self, query: str) -> str:
        """Answer a question based on the PDF content."""
        try:
            # Find most relevant chunk
            query_embedding = self.embedding_model.encode([query])
            print(query_embedding)
            similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
            top_idx = np.argmax(similarities)
            print("Printingggg",similarities)
            
            # Generate response using Llama
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Answer based on the provided context."},
                    {"role": "user", "content": f"Answer the following question based solely on the given context. Do not add any information beyond what is presented here.\n\nQuestion: {query}\nContext: {self.text_chunks[top_idx]}"}
                ],
                temperature=0.1,
                max_tokens=0
            )

            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return f"Error processing question: {str(e)}"

# Define the /ask route
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data['question']
        logger.info(f"Received question: {question}")
        
        # Get the answer
        answer = qa_system.answer_question(question)
        return jsonify({'answer': answer})
    
    except Exception as e:
        logger.error(f"Error in /ask route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Initialize the QA system with the model and PDF paths
    qa_system = PDFQuestionAnswering(
        model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
        pdf_path="./docs/IPC.pdf"
    )
    app.run(debug=True)
