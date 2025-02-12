import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from db_helper import ContextualRetrieval

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_SECRET_KEY')
os.environ["ANTHROPIC_API_KEY"]  = os.getenv('CLAUDE_KEY')

def generate_response(retriever, query,model_provider = "openai",
                      model_name = "gpt-4o-mini"):
    if model_provider.lower() == "anthropic":
        model = ChatAnthropic(model=model_name,temperature=0,max_tokens=1024,timeout=None,max_retries=2)
    else:
        model = ChatOpenAI(model=model_name, temperature=0, max_tokens=1024, timeout=None, max_retries=2)

    chain = RetrievalQAWithSourcesChain.from_chain_type(model,
                                                        chain_type="stuff",
                                                        retriever=retriever
                                                        )
    result = chain.invoke({"question": query},return_only_outputs=True)

    return result




# Question = []
# cr = ContextualRetrieval()
# neo4j_index = cr.load_neo4j_index(index_name="lectureNoteContextualIndex")
# retriever = neo4j_index.as_retriever()