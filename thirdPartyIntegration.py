#Documentation: 
# Neo4j (https://neo4j.com/developer-blog/neo4j-langchain-vector-index-implementation/)
# Pinecone (https://docs.pinecone.io/integrations/langchain)
#----------------------------------Import Libraries-----------------------------------------
#Environmental Variable
import os
from dotenv import load_dotenv

#langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain.docstore.document import Document #Document class from the langchain.docstore.document module.
from langchain_anthropic import ChatAnthropic

#others
from typing import List, Union #List type hint from the typing module

#-------------------------------------------------------------------------------------------

#----------------------------Variable Declaration and Initialization------------------------

#***********Env variable******************
load_dotenv()
OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')
NEO4J_KEY = os.getenv('NEO4J_KEY')
CLAUDE_KEY = os.getenv('CLAUDE_KEY')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
#******************************************

openAIEmbeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
index_name = "ed_ai"
claude_model = ChatAnthropic(model='claude-3-opus-20240229', temperature=0, anthropic_api_key=CLAUDE_KEY)
openai_model = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0, api_key=OPEN_AI_KEY)
#-------------------------------------------------------------------------------------------

#--------------------------------Function Declaration---------------------------------------

#*************** Neo4j ********************************
def create_neo4j_vectordb(documents: List[Document]) -> Neo4jVector:
    """
        Creates a Neo4j vector database from a list of documents.

        This function takes a list of documents and uses the OpenAI Embeddings API
        to generate vector representations for each document. It then creates a
        Neo4j database instance and populates it with nodes representing the documents
        and their corresponding vectors.

        Args:
            documents (List[Document]): A list of documents to be processed.
                Each document should have a text property that contains the content
                to be embedded.

        Returns:
            Neo4jVector: An instance of the Neo4jVector class representing the
                created Neo4j vector database.

        Raises:
            Exception: If there is an error creating the Neo4j database or
                embedding the documents.

        **Optional Parameters (default values shown):**

        * url (str): The URL of the Neo4j database server. Defaults to `NEO4J_URI`
            environment variable.
        * username (str): The username for accessing the Neo4j database. Defaults
            to `NEO4J_USERNAME` environment variable.
        * password (str): The password for accessing the Neo4j database. Defaults
            to `NEO4J_PASSWORD` environment variable.
        * database (str): The name of the Neo4j database to use. Defaults to
            "neo4j_ed".
        * index_name (str): The name of the index to create for efficient vector
            search. Defaults to "ed_ai".
        * node_label (str): The label to assign to nodes representing documents.
            Defaults to "edResources".
        * text_node_property (str): The name of the property to store the document
            text on nodes. Defaults to "content".
        * embedding_node_property (str): The name of the property to store the
            embedded vector on nodes. Defaults to "vector".
        * create_id_index (bool): Whether to create an index on the document IDs
            for faster lookups. Defaults to True.
  """
    # Instantiate Neo4j vector from documents
    neo4j_vector = Neo4jVector.from_documents(
        documents,
        openAIEmbeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database="neo4j_ed",  # neo4j by default, neo4j_ed
        index_name= index_name,  # vector by default
        node_label="edResources",  # Chunk by default
        text_node_property="content",  # text by default | info
        embedding_node_property="vector",  # embedding by default
        create_id_index=True,  # True by default
    )
    return neo4j_vector

def update_neo4j_vectordb(documents: List[Document] = None, mode: int = 0) -> Union[Neo4jVector, Exception]:
    """
        Updates a Neo4j vector database with new documents.

        This function attempts to connect to an existing Neo4j vector database based
        on the provided configuration (URL, username, password, index name, etc.). If
        the database and index exist, it adds the new documents (represented in the
        `documents` list) to the database.

        If the database or index cannot be found, the function attempts to create a
        new Neo4j vector database using the `create_neo4j_vectordb` function and then
        adds the documents.

        Args:
            documents (List[Document]): A list of documents to be added to the
                Neo4j vector database. Each document should have a text property
                containing the content to be embedded.

        Returns:
            Union[Neo4jVector, Exception]:
                - If successful, returns an instance of the `Neo4jVector` class
                    representing the updated Neo4j vector database.
                - If there is an error, returns the encountered Exception object.

        **Optional Parameters (inherited from create_neo4j_vectordb):**

        The function inherits the optional parameters documented in the
        `create_neo4j_vectordb` function, which specify details about the Neo4j
        connection and database configuration. These parameters are used when
        creating a new database if the existing one is not found.
    """
    #check if there is an existing index
    try:
        neo4j_db = Neo4jVector.from_existing_index(openAIEmbeddings, url=NEO4J_URI,
                                                username=NEO4J_USERNAME,
                                                password=NEO4J_PASSWORD, 
                                                index_name="ed_ai",
                                                text_node_property="content",)
        #if the db is just requested
        if mode:
            return neo4j_db
    except Exception as e:
        print(e)
        neo4j_db = None
         
    #if there is
    if neo4j_db:
        neo4j_db.add_documents(documents)
    else:
        try:
            neo4j_db = create_neo4j_vectordb(documents)
        except Exception as e:
            return e
    return neo4j_db
#******************************************************

#*************** PineCone *****************************
def create_pinecone_vectordb(documents: List[Document]) -> PineconeVectorStore:
    """
        Creates a Pinecone vector database from a list of documents.

        This function takes a list of documents and uses the provided OpenAI Embeddings
        API to generate vector representations for each document. It then creates a
        Pinecone vector database instance and populates it with the document vectors.

        Args:
            documents (List[Document]): A list of documents to be processed.
                Each document should have a text property that contains the content
                to be embedded.

        Returns:
            PineconeVectorStore: An instance of the PineconeVectorStore class representing
                the created Pinecone vector database.

        Raises:
            Exception: If there is an error creating the Pinecone vector database or
                embedding the documents.

        **Optional Parameter:**

        * index_name (str, optional): The name of the index to create for efficient
            vector search. Defaults to the implementation's default index name.
    """
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        documents,
        index_name=index_name,
        embedding=openAIEmbeddings)
    return vectorstore_from_docs

def update_pinecone_vectordb(documents: List[Document], mode: int = 0) -> Union[PineconeVectorStore, Exception]:
    """
        Updates a Pinecone vector database with new documents.

        This function attempts to connect to an existing Pinecone vector database 
        using the provided index name and OpenAI Embeddings configuration. 
        If the database exists, it adds the new documents (represented in the 
        `documents` list) to the database.

        If the database cannot be found, the function attempts to create a new 
        Pinecone vector database using the `create_pinecone_vectordb` function and then 
        adds the documents.

        Args:
            documents (List[Document]): A list of documents to be added to the
                Pinecone vector database. Each document should have a text property
                containing the content to be embedded.

        Returns:
            Union[PineconeVectorStore, Exception]:
                - If successful, returns an instance of the `PineconeVectorStore` class
                    representing the updated Pinecone vector database.
                - If there is an error, returns the encountered Exception object.

        **Optional Parameter:**

        * index_name (str, optional): The name of the index to use for efficient
            vector search. Defaults to the implementation's default index name.
    """
    try:
        #Check if index exists
        pinecone_db = PineconeVectorStore(index_name=index_name, embedding=openAIEmbeddings)

        #if the db is just requested
        if mode:
            return pinecone_db

    except Exception as e:
        print(e)
        pinecone_db = None
            
    #if there is
    if pinecone_db:
        pinecone_db.add_documents(documents)
    else:
        try:
            pinecone_db = create_pinecone_vectordb(documents)
        except Exception as e:
            return e
    return pinecone_db
#******************************************************

