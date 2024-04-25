#----------------------------------Import Libraries-----------------------------------------
#langchain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain

#others
from typing import List, Union #List type hint from the typing module
from thirdPartyIntegration import claude_model, openai_model

#-------------------------------------------------------------------------------------------

def vector_search(vectordb, query,db = "neo4j"):
    if db.lower() == "neo4j":
        results = vectordb.similarity_search(query, k=4)
        return results
    elif db.lower() == "pinecone":
        results = vectordb.similarity_search(query, k=4)
        return results

def knowledge_graph():
    pass

def generate_response(vectordb, query,model_name = "anthropic"):
    if model_name.lower() == "anthropic":
        model = claude_model
    else:
        model = openai_model

    chain = RetrievalQAWithSourcesChain.from_chain_type(model,
                                                        chain_type="map_reduce",
                                                        retriever=vectordb.as_retriever()
                                                        )
    result = chain({"question": query},return_only_outputs=True)

    return result