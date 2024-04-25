from ingest import process_documents, store_document

# from dotenv import load_dotenv
# import os

# from langchain_community.vectorstores import Neo4jVector
# from langchain.embeddings.openai import OpenAIEmbeddings

# load_dotenv()
# OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')
# NEO4J_KEY = os.getenv('NEO4J_KEY')
# CLAUDE_KEY = os.getenv('CLAUDE_KEY')
# NEO4J_URI = os.getenv('NEO4J_URI')
# NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
# NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
# #******************************************

# openAIEmbeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)

def main():
    docs = process_documents("data", is_dir=True)

    store_document(docs)

if __name__ == "__main__":
    main()