# RAG Chatbot for AI Literacy

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

### 2. Install dependencies
pip install -r requirements.txt

### 3. Set up environment variables
cp .env.example .env

# Then edit .env and add your API keys:
# - Get OpenAI key: https://platform.openai.com/api-keys
# - Get Cohere key: https://dashboard.cohere.com/api-keys
# - Get LangSmith key (optional): https://smith.langchain.com/settings

### 4. Add your documents
Place your PDF files in `rag/data_ingestion/`

### 5. Build the vector database
python rag/build_db.py

### 6. Run the app
streamlit run frontend/main.py

## API Costs (might change)
- OpenAI (GPT-4o-mini): ~$0.15 per 1M tokens
- Cohere (Rerank): Free for first 10K requests/month

**Note: You are responsible for your own API costs.**