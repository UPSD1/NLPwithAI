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
from neo4j import GraphDatabase
import uuid
import hashlib
from openai import OpenAI

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

NEO4J_DATABASE = "neo4j"
EMBEDDING_MODEL = "text-embedding-ada-002"
index_name = "ed_ai"

openAIEmbeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
claude_model = ChatAnthropic(model='claude-3-opus-20240229', temperature=0, anthropic_api_key=CLAUDE_KEY)
openai_model = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0, api_key=OPEN_AI_KEY)
#-------------------------------------------------------------------------------------------

#--------------------------------Function Declaration---------------------------------------

#*************** Neo4j ********************************
def create_neo4j_vectordb(documents: List[Document], index_name: str, node_label:str) -> Neo4jVector:
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
        database="neo4j",  # neo4j by default, neo4j_ed
        index_name= index_name,  # vector by default
        node_label=node_label,  # Chunk by default
        text_node_property="info",  # text by default
        embedding_node_property="vector",  # embedding by default
        create_id_index=True,  # True by default

    )
    return neo4j_vector

def update_neo4j_vectordb(documents: List[Document] = None, mode: int = 0, index_name:str = "vector", node_label:str = "chunk") -> Union[Neo4jVector, Exception]:
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
                                                database="neo4j",
                                                index_name=index_name,
                                                node_label=node_label,
                                                text_node_property="info",
                                                )
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
            neo4j_db = create_neo4j_vectordb(documents, index_name, node_label)
        except Exception as e:
            return e
    return neo4j_db

def initialiseNeo4j():
    cypher_schema = [
        "CREATE CONSTRAINT sectionKey IF NOT EXISTS FOR (c:Section) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT chunkKey IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT documentKey IF NOT EXISTS FOR (c:Document) REQUIRE (c.url_hash) IS UNIQUE;",
        "CREATE CONSTRAINT tableKey IF NOT EXISTS FOR (c:Table) REQUIRE (c.key) IS UNIQUE;",
        "CALL db.index.vector.createNodeIndex('chunkVectorIndex', 'Embedding', 'value', 1536, 'COSINE');"
    ]

    driver = GraphDatabase.driver(NEO4J_URI, database=NEO4J_DATABASE, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        for cypher in cypher_schema:
            session.run(cypher)

    driver.close()

def ingestDocumentNeo4j(doc, doc_location):
    cypher_pool = [
        # 0 - Document
        "MERGE (d:Document {url_hash: $doc_url_hash_val}) ON CREATE SET d.url = $doc_url_val RETURN d;",  
        # 1 - Section
        "MERGE (p:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) ON CREATE SET p.page_idx = $page_idx_val, p.title_hash = $title_hash_val, p.block_idx = $block_idx_val, p.title = $title_val, p.tag = $tag_val, p.level = $level_val RETURN p;",
        # 2 - Link Section with the Document
        "MATCH (d:Document {url_hash: $doc_url_hash_val}) MATCH (s:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (d)<-[:HAS_DOCUMENT]-(s);",
        # 3 - Link Section with a parent section
        "MATCH (s1:Section {key: $doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_title_hash_val}) MATCH (s2:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (s1)<-[:UNDER_SECTION]-(s2);",
        # 4 - Chunk
        "MERGE (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) ON CREATE SET c.sentences = $sentences_val, c.sentences_hash = $sentences_hash_val, c.block_idx = $block_idx_val, c.page_idx = $page_idx_val, c.tag = $tag_val, c.level = $level_val RETURN c;",
        # 5 - Link Chunk to Section
        "MATCH (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) MATCH (s:Section {key:$doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(c);",
        # 6 - Table
        "MERGE (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) ON CREATE SET t.name = $name_val, t.doc_url_hash = $doc_url_hash_val, t.block_idx = $block_idx_val, t.page_idx = $page_idx_val, t.html = $html_val, t.rows = $rows_val RETURN t;",
        # 7 - Link Table to Section
        "MATCH (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Section {key: $doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);",
        # 8 - Link Table to Document if no parent section
        "MATCH (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Document {url_hash: $doc_url_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);"
    ]

    driver = GraphDatabase.driver(NEO4J_URI, database=NEO4J_DATABASE, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        cypher = ""

        # 1 - Create Document node
        doc_url_val = doc_location
        doc_url_hash_val = hashlib.md5(doc_url_val.encode("utf-8")).hexdigest()

        cypher = cypher_pool[0]
        session.run(cypher, doc_url_hash_val=doc_url_hash_val, doc_url_val=doc_url_val)

        # 2 - Create Section nodes
        
        countSection = 0
        for sec in doc.sections():
            sec_title_val = sec.title
            sec_title_hash_val = hashlib.md5(sec_title_val.encode("utf-8")).hexdigest()
            sec_tag_val = sec.tag
            sec_level_val = sec.level
            sec_page_idx_val = sec.page_idx
            sec_block_idx_val = sec.block_idx

            # MERGE section node
            if not sec_tag_val == 'table':
                cypher = cypher_pool[1]
                session.run(cypher, page_idx_val=sec_page_idx_val
                                , title_hash_val=sec_title_hash_val
                                , title_val=sec_title_val
                                , tag_val=sec_tag_val
                                , level_val=sec_level_val
                                , block_idx_val=sec_block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )

                # Link Section with a parent section or Document

                sec_parent_val = str(sec.parent.to_text())

                if sec_parent_val == "None":    # use Document as parent

                    cypher = cypher_pool[2]
                    session.run(cypher, page_idx_val=sec_page_idx_val
                                    , title_hash_val=sec_title_hash_val
                                    , doc_url_hash_val=doc_url_hash_val
                                    , block_idx_val=sec_block_idx_val
                                )

                else:   # use parent section
                    sec_parent_title_hash_val = hashlib.md5(sec_parent_val.encode("utf-8")).hexdigest()
                    sec_parent_page_idx_val = sec.parent.page_idx
                    sec_parent_block_idx_val = sec.parent.block_idx

                    cypher = cypher_pool[3]
                    session.run(cypher, page_idx_val=sec_page_idx_val
                                    , title_hash_val=sec_title_hash_val
                                    , block_idx_val=sec_block_idx_val
                                    , parent_page_idx_val=sec_parent_page_idx_val
                                    , parent_title_hash_val=sec_parent_title_hash_val
                                    , parent_block_idx_val=sec_parent_block_idx_val
                                    , doc_url_hash_val=doc_url_hash_val
                                )
            # **** if sec_parent_val == "None":    

            countSection += 1
        # **** for sec in doc.sections():

        
        # ------- Continue within the blocks -------
        # 3 - Create Chunk nodes from chunks
            
        countChunk = 0
        for chk in doc.chunks():

            chunk_block_idx_val = chk.block_idx
            chunk_page_idx_val = chk.page_idx
            chunk_tag_val = chk.tag
            chunk_level_val = chk.level
            chunk_sentences = "\n".join(chk.sentences)

            # MERGE Chunk node
            if not chunk_tag_val == 'table':
                chunk_sentences_hash_val = hashlib.md5(chunk_sentences.encode("utf-8")).hexdigest()

                # MERGE chunk node
                cypher = cypher_pool[4]
                session.run(cypher, sentences_hash_val=chunk_sentences_hash_val
                                , sentences_val=chunk_sentences
                                , block_idx_val=chunk_block_idx_val
                                , page_idx_val=chunk_page_idx_val
                                , tag_val=chunk_tag_val
                                , level_val=chunk_level_val
                                , doc_url_hash_val=doc_url_hash_val
                            )
            
                # Link chunk with a section
                # Chunk always has a parent section 

                chk_parent_val = str(chk.parent.to_text())
                
                if not chk_parent_val == "None":
                    chk_parent_hash_val = hashlib.md5(chk_parent_val.encode("utf-8")).hexdigest()
                    chk_parent_page_idx_val = chk.parent.page_idx
                    chk_parent_block_idx_val = chk.parent.block_idx

                    cypher = cypher_pool[5]
                    session.run(cypher, sentences_hash_val=chunk_sentences_hash_val
                                    , block_idx_val=chunk_block_idx_val
                                    , parent_hash_val=chk_parent_hash_val
                                    , parent_block_idx_val=chk_parent_block_idx_val
                                    , doc_url_hash_val=doc_url_hash_val
                                )
                    
                # Link sentence 
                #   >> TO DO for smaller token length

                countChunk += 1
        # **** for chk in doc.chunks(): 

        # 4 - Create Table nodes

        countTable = 0
        for tb in doc.tables():
            page_idx_val = tb.page_idx
            block_idx_val = tb.block_idx
            name_val = 'block#' + str(block_idx_val) + '_' + tb.name
            html_val = tb.to_html()
            rows_val = len(tb.rows)

            # MERGE table node

            cypher = cypher_pool[6]
            session.run(cypher, block_idx_val=block_idx_val
                            , page_idx_val=page_idx_val
                            , name_val=name_val
                            , html_val=html_val
                            , rows_val=rows_val
                            , doc_url_hash_val=doc_url_hash_val
                        )
            
            # Link table with a section
            # Table always has a parent section 

            table_parent_val = str(tb.parent.to_text())
            
            if not table_parent_val == "None":
                table_parent_hash_val = hashlib.md5(table_parent_val.encode("utf-8")).hexdigest()
                table_parent_page_idx_val = tb.parent.page_idx
                table_parent_block_idx_val = tb.parent.block_idx

                cypher = cypher_pool[7]
                session.run(cypher, name_val=name_val
                                , block_idx_val=block_idx_val
                                , parent_page_idx_val=table_parent_page_idx_val
                                , parent_hash_val=table_parent_hash_val
                                , parent_block_idx_val=table_parent_block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )

            else:   # link table to Document
                cypher = cypher_pool[8]
                session.run(cypher, name_val=name_val
                                , block_idx_val=block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )
            countTable += 1

        # **** for tb in doc.tables():
        
        print(f'\'{doc_url_val}\' Done! Summary: ')
        print('#Sections: ' + str(countSection))
        print('#Chunks: ' + str(countChunk))
        print('#Tables: ' + str(countTable))

    driver.close()
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

#**************** OpenAI ******************************
def get_embedding(client, text, model):
    response = client.embeddings.create(
                    input=text,
                    model=model,
                )
    return response.data[0].embedding

def LoadEmbedding(label, property):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    openai_client = OpenAI (api_key = OPEN_AI_KEY)

    with driver.session() as session:
        # get chunks in document, together with their section titles
        result = session.run(f"MATCH (ch:{label}) -[:HAS_PARENT]-> (s:Section) RETURN id(ch) AS id, s.title + ' >> ' + ch.{property} AS text")
        # call OpenAI embedding API to generate embeddings for each proporty of node
        # for each node, update the embedding property
        count = 0
        for record in result:
            id = record["id"]
            text = record["text"]

            # For better performance, text can be batched
            embedding = get_embedding(openai_client, text, EMBEDDING_MODEL)

            # key property of Embedding node differentiates different embeddings
            cypher = "CREATE (e:Embedding) SET e.key=$key, e.value=$embedding"
            cypher = cypher + " WITH e MATCH (n) WHERE id(n) = $id CREATE (n) -[:HAS_EMBEDDING]-> (e)"
            session.run(cypher,key=property, embedding=embedding, id=id )
            count = count + 1

        session.close()

        print("Processed " + str(count) + " " + label + " nodes for property @" + property + ".")
        return count

def load_embedding_model():
    embeddings = openAIEmbeddings
    dimension = 1536
    return embeddings, dimension
#*******************************************************

#**************** LLMs **********************************
def load_llm(llm_name: str):

    if llm_name == "gpt-4":
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True, openai_api_key = OPEN_AI_KEY)
    elif llm_name == "claude":
        return ChatAnthropic(model='claude-3-opus-20240229', temperature=0.3, anthropic_api_key=CLAUDE_KEY)
    
    #default use is claude
    return ChatAnthropic(model='claude-3-opus-20240229', temperature=0.3, anthropic_api_key=CLAUDE_KEY)
#******************************************************