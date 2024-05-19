from edIntegration import authenticate, ed
from thirdPartyIntegration import update_neo4j_vectordb, load_llm, load_embedding_model
from search import vector_search, generate_response, configure_llm_only_chain, configure_qa_structure_rag_chain
from langchain_community.graphs import Neo4jGraph
import time

#Environmental Variable
import os
from dotenv import load_dotenv

load_dotenv()
OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')
NEO4J_KEY = os.getenv('NEO4J_KEY')
CLAUDE_KEY = os.getenv('CLAUDE_KEY')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = "neo4j"

llm_name = "claude"
course_id = 58877

def setup():
    embeddings, dimension = load_embedding_model()

    llm = load_llm(llm_name)

    # llm_chain: LLM only response
    llm_chain = configure_llm_only_chain(llm)

    # rag_chain: KG augmented response
    rag_chain = configure_qa_structure_rag_chain(
        llm, embeddings, embeddings_store_url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
    )

    return rag_chain

# comment = ed.post_comment(4966179, param)

def main():
    status, _= authenticate(verbose=False)
    if status: #if authentication was successful
        print("logged in")

    # #get already existing db
    # vectordb = update_neo4j_vectordb(mode=1)

    # #Semantic search
    # query = "What's the Ultimate Goal?"
    # result = vector_search(vectordb, query)
    # print(result)

    # response = generate_response(vectordb, query)
    # print(response)

    #load model
    model = setup()
    #query the KG and Vector DB
    # details = ed.get_thread(4947717)
    # result = model.invoke({"question": details['document']})["answer"]
    # print(f"\nquery > {details['document']}")
    # print(f"AI > {result}")
    # ed.post_comment(4947717, result)
    
    # while(True):
    #     print("Enter your query below")
    #     user_input = input()
    #     if (user_input.lower().strip() == 'exit') or (user_input.lower().strip() == 'quit'):
    #         break
    #     start = time.time()
    #     result = model.invoke({"question": user_input})["answer"]
    #     print(f"\nquery > {user_input}")
    #     print(f"AI > {result}")
    #     print()
    #     print(f"{time.time() - start: 0.4f}secs used\n")

    # thread_lst = ed.list_threads(course_id, limit = 30, offset = 0, sort = "new" )

    # for i in range(len(thread_lst)):
    #     if(thread_lst[i]['id'] != 4947717):
    #         details = ed.get_thread(thread_lst[i]['id'])
    #         result = model.invoke({"question": details['document']})["answer"]
    #         ed.post_comment(thread_lst[i]['id'], result)
    
    # print("Question Done")
    counter = 0
    latest_id = 4952434
    while(True):
        thread_lst = ed.list_threads(course_id, limit = 30, offset = 0, sort = "new" )
        # counter += 1
        # print(counter)
        if(thread_lst[0]['id'] != latest_id):
            print("found new question")
            details = ed.get_thread(thread_lst[0]['id'])
            result = model.invoke({"question": details['document']})["answer"]
            ed.post_comment(thread_lst[0]['id'], result)
            latest_id = thread_lst[0]['id']
            print("Answered latest question")
        time.sleep(1)
        



if __name__ == "__main__":
    main()