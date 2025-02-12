from ingest import process_documents, store_document, load_graph_pdf
from thirdPartyIntegration import initialiseNeo4j, LoadEmbedding

def main():
    #vectorDB
    # docs = process_documents("data", is_dir=True)
    # store_document(docs)

    #graphDB
    #initiailize
    # initialiseNeo4j()
    #populate with Natural language text
    # load_graph_pdf()
    #load embedding into graphdb
    LoadEmbedding("Chunk", "sentences")
    LoadEmbedding("Table", "name")


if __name__ == "__main__":
    main()