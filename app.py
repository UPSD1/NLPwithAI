from edIntegration import authenticate
from thirdPartyIntegration import update_neo4j_vectordb
from search import vector_search, generate_response

def main():
    status, _= authenticate(verbose=False)
    if status: #if authentication was successful
        print("logged in")

    vectordb = update_neo4j_vectordb(mode=1)

    query = "What's the Ultimate Goal?"
    result = vector_search(vectordb, query)
    print(result)

    response = generate_response(vectordb, query)
    print(response)



if __name__ == "__main__":
    main()