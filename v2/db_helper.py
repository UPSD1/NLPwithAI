#-------------------------------------LIBRARY-------------------------------------------
import os
import glob
import hashlib
import logging
from typing import List
from datetime import datetime
from neo4j import GraphDatabase
from llmsherpa.readers import LayoutPDFReader  # Assumes this module exists
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import (
    CSVLoader, PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, Docx2txtLoader, UnstructuredURLLoader
)
from dotenv import load_dotenv
#---------------------------------------------------------------------------------------

#-------------------------Variables, Configuration, initialization----------------------
# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_SECRET_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xls": (UnstructuredExcelLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    "url": (UnstructuredURLLoader, {}),
}
#----------------------------------------------------------------------------------------

#--------------------------------CLASSESS------------------------------------------------
class Neo4jManager:
    """
    Manage connections and operations with the Neo4j database.
    """
    def __init__(self, uri=os.environ["NEO4J_URI"], user="neo4j",
                 password=os.environ["NEO4J_PASSWORD"], database="neo4j",
                 debug = False):
        """
        Initialize the Neo4jManager with connection details.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.debug = debug
        self._connect()

    def _connect(self):
        """
        Establish a connection to the Neo4j database.
        """
        try:
            if self.debug: print("Connecting to Neo4j...")
            logging.info("Connecting to Neo4j...")
            self.driver = GraphDatabase.driver(self.uri, database=self.database, 
                                               auth=(self.user, self.password))
            if self.debug: print("Connected to Neo4j successfully.")
            logging.info("Connected to Neo4j successfully.")
        except Exception as e:
            logging.error("Failed to connect to Neo4j: %s", e)
            if self.debug: print("Failed to connect to Neo4j: %s", e)
            raise

    def initialise(self, index_name='vector', vector_dimension=1536):
        """
        Initialize the Neo4j database by creating constraints and a vector index.
        
        Args:
            index_name (str): Name of the vector index. Defaults to 'vector'.
            vector_dimension (int): Dimensionality of the vector. Defaults to 1536.
        
        Raises:
            Exception: Propagates any errors during Cypher execution.
        """
        # List of Cypher commands for constraints and index
        cypher_schema = [
            # Ensure the 'key' property is unique for all nodes with the 'Section' label
            "CREATE CONSTRAINT sectionKey IF NOT EXISTS FOR (c:Section) REQUIRE (c.key) IS UNIQUE;",
            # Ensure the 'key' property is unique for all nodes with the 'Chunk' label
            "CREATE CONSTRAINT chunkKey IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.key) IS UNIQUE;",
            # Ensure the 'url_hash' property is unique for all nodes with the 'Document' label
            "CREATE CONSTRAINT documentKey IF NOT EXISTS FOR (c:Document) REQUIRE (c.url_hash) IS UNIQUE;",
            # Ensure the 'key' property is unique for all nodes with the 'Table' label
            "CREATE CONSTRAINT tableKey IF NOT EXISTS FOR (c:Table) REQUIRE (c.key) IS UNIQUE;",
            # Create a vector index with the specified name and dimensions on the 'value' property of nodes with the 'Embedding' label
            f"CALL db.index.vector.createNodeIndex('{index_name}', 'Embedding', 'value', {vector_dimension}, 'COSINE');"
        ]
        try:
            with self.driver.session() as session:
                for cypher in cypher_schema:
                    if self.debug: print("Executing Cypher: %s", cypher)
                    logging.debug("Executing Cypher: %s", cypher)
                    session.run(cypher)
            logging.info("Neo4j initialization complete.")
            if self.debug: print("Neo4j initialization complete.")
        except Exception as e:
            if self.debug: print("Error during Neo4j initialization: %s", e)
            logging.error("Error during Neo4j initialization: %s", e)
            raise

    def ingest_document(self, doc, doc_location):
        """
        Ingest a parsed document into the Neo4j database by creating nodes and relationships.
        
        Args:
            doc (Document): The parsed document with sections, chunks, and tables.
            doc_location (str): The location (path) of the document.
        
        Raises:
            Exception: Propagates any errors during document ingestion.
        """
        # List of Cypher queries to be executed during ingestion
        cypher_pool = [
            # Create or match Document node
            "MERGE (d:Document {name: $doc_name_val}) ON CREATE SET d.url = $doc_url_val RETURN d;",
            # Create or match Section node
            "MERGE (p:Section {key: $doc_name_val+'|'+$block_idx_val+'|'+$title_hash_val}) ON CREATE SET p.page_idx = $page_idx_val, p.title_hash = $title_hash_val, p.block_idx = $block_idx_val, p.title = $title_val, p.tag = $tag_val, p.level = $level_val RETURN p;",
            # Link Section to Document
            "MATCH (d:Document {name: $doc_name_val}) MATCH (s:Section {key: $doc_name_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (d)<-[:HAS_DOCUMENT]-(s);",
            # Link parent Section to child Section
            "MATCH (s1:Section {key: $doc_name_val+'|'+$parent_block_idx_val+'|'+$parent_title_hash_val}) MATCH (s2:Section {key: $doc_name_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (s1)<-[:UNDER_SECTION]-(s2);",
            # Create or match Chunk node
            "MERGE (c:Chunk {key: $doc_name_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) ON CREATE SET c.sentences = $sentences_val, c.sentences_hash = $sentences_hash_val, c.block_idx = $block_idx_val, c.page_idx = $page_idx_val, c.tag = $tag_val, c.level = $level_val RETURN c;",
            # Link Chunk to Section
            "MATCH (c:Chunk {key: $doc_name_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) MATCH (s:Section {key:$doc_name_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(c);",
            # Create or match Table node
            "MERGE (t:Table {key: $doc_name_val+'|'+$block_idx_val+'|'+$name_val}) ON CREATE SET t.name = $name_val, t.doc_name = $doc_name_val, t.block_idx = $block_idx_val, t.page_idx = $page_idx_val, t.html = $html_val, t.rows = $rows_val RETURN t;",
            # Link Table to Section
            "MATCH (t:Table {key: $doc_name_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Section {key: $doc_name_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);",
            # Link Table to Document
            "MATCH (t:Table {key: $doc_name_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Document {name: $doc_name_val}) MERGE (s)<-[:HAS_PARENT]-(t);"
        ]
        try:
            with self.driver.session() as session:
                # Derive document name and URL
                doc_name_val = os.path.basename(doc_location)
                doc_url_val = doc_location
                if self.debug: print("Ingesting Document: %s", doc_name_val)
                logging.info("Ingesting Document: %s", doc_name_val)
                # Create the Document node
                session.run(cypher_pool[0], doc_name_val=doc_name_val, doc_url_val=doc_url_val)

                # Process each Section in the document
                for sec in doc.sections():
                    sec_title_val = sec.title
                    sec_title_hash_val = hashlib.md5(sec_title_val.encode("utf-8")).hexdigest()
                    sec_tag_val = sec.tag
                    sec_level_val = sec.level
                    sec_page_idx_val = sec.page_idx
                    sec_block_idx_val = sec.block_idx

                    if sec_tag_val != 'table':
                        if self.debug: print("Processing Section: %s", sec_title_val)
                        logging.debug("Processing Section: %s", sec_title_val)
                        session.run(cypher_pool[1],
                                    page_idx_val=sec_page_idx_val,
                                    title_hash_val=sec_title_hash_val,
                                    title_val=sec_title_val,
                                    tag_val=sec_tag_val,
                                    level_val=sec_level_val,
                                    block_idx_val=sec_block_idx_val,
                                    doc_name_val=doc_name_val)
                        sec_parent_val = str(sec.parent.to_text())
                        if sec_parent_val == "None":
                            session.run(cypher_pool[2],
                                        page_idx_val=sec_page_idx_val,
                                        title_hash_val=sec_title_hash_val,
                                        doc_name_val=doc_name_val,
                                        block_idx_val=sec_block_idx_val)
                        else:
                            sec_parent_title_hash_val = hashlib.md5(sec_parent_val.encode("utf-8")).hexdigest()
                            sec_parent_page_idx_val = sec.parent.page_idx
                            sec_parent_block_idx_val = sec.parent.block_idx
                            session.run(cypher_pool[3],
                                        page_idx_val=sec_page_idx_val,
                                        title_hash_val=sec_title_hash_val,
                                        block_idx_val=sec_block_idx_val,
                                        parent_page_idx_val=sec_parent_page_idx_val,
                                        parent_title_hash_val=sec_parent_title_hash_val,
                                        parent_block_idx_val=sec_parent_block_idx_val,
                                        doc_name_val=doc_name_val)

                # Process each Chunk in the document
                for chk in doc.chunks():
                    chunk_block_idx_val = chk.block_idx
                    chunk_page_idx_val = chk.page_idx
                    chunk_tag_val = chk.tag
                    chunk_level_val = chk.level
                    chunk_sentences = "\n".join(chk.sentences)
                    if chunk_tag_val != 'table':
                        chunk_sentences_hash_val = hashlib.md5(chunk_sentences.encode("utf-8")).hexdigest()
                        if self.debug: print("Processing Chunk in block: %s", chunk_block_idx_val)
                        logging.debug("Processing Chunk in block: %s", chunk_block_idx_val)
                        session.run(cypher_pool[4],
                                    sentences_hash_val=chunk_sentences_hash_val,
                                    sentences_val=chunk_sentences,
                                    block_idx_val=chunk_block_idx_val,
                                    page_idx_val=chunk_page_idx_val,
                                    tag_val=chunk_tag_val,
                                    level_val=chunk_level_val,
                                    doc_name_val=doc_name_val)
                        chk_parent_val = str(chk.parent.to_text())
                        if chk_parent_val != "None":
                            chk_parent_hash_val = hashlib.md5(chk_parent_val.encode("utf-8")).hexdigest()
                            chk_parent_page_idx_val = chk.parent.page_idx
                            chk_parent_block_idx_val = chk.parent.block_idx
                            session.run(cypher_pool[5],
                                        sentences_hash_val=chunk_sentences_hash_val,
                                        block_idx_val=chunk_block_idx_val,
                                        parent_hash_val=chk_parent_hash_val,
                                        parent_block_idx_val=chk_parent_block_idx_val,
                                        doc_name_val=doc_name_val)

                # Process each Table in the document
                for tb in doc.tables():
                    page_idx_val = tb.page_idx
                    block_idx_val = tb.block_idx
                    name_val = 'block#' + str(block_idx_val) + '_' + tb.name
                    html_val = tb.to_html()  # Convert table to HTML
                    rows_val = len(tb.rows)
                    if self.debug: print("Processing Table: %s", name_val)
                    logging.debug("Processing Table: %s", name_val)
                    session.run(cypher_pool[6],
                                block_idx_val=block_idx_val,
                                page_idx_val=page_idx_val,
                                name_val=name_val,
                                html_val=html_val,
                                rows_val=rows_val,
                                doc_name_val=doc_name_val)
                    table_parent_val = str(tb.parent.to_text())
                    if table_parent_val != "None":
                        table_parent_hash_val = hashlib.md5(table_parent_val.encode("utf-8")).hexdigest()
                        table_parent_page_idx_val = tb.parent.page_idx
                        table_parent_block_idx_val = tb.parent.block_idx
                        session.run(cypher_pool[7],
                                    name_val=name_val,
                                    block_idx_val=block_idx_val,
                                    parent_page_idx_val=table_parent_page_idx_val,
                                    parent_hash_val=table_parent_hash_val,
                                    parent_block_idx_val=table_parent_block_idx_val,
                                    doc_name_val=doc_name_val)
                    else:
                        session.run(cypher_pool[8],
                                    name_val=name_val,
                                    block_idx_val=block_idx_val,
                                    doc_name_val=doc_name_val)
                logging.info("Document ingestion complete: %s", doc_name_val)
                if self.debug: print("Document ingestion complete: %s", doc_name_val)

                if self.debug:
                    # Print summary after document ingestion is complete
                    print(f'\'{doc_name_val}\' Done! Summary: ')
                    print('#Sections: ' + str(len(doc.sections())))
                    print('#Chunks: ' + str(len(doc.chunks())))
                    print('#Tables: ' + str(len(doc.tables())))

        except Exception as e:
            if self.debug: print("Error during document ingestion: %s", e)
            logging.error("Error during document ingestion: %s", e)
            raise

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        if self.driver:
            self.driver.close()
            if self.debug: print("Neo4j driver closed.")
            logging.info("Neo4j driver closed.")

class PDFIngestor:
    """
    Parse PDF files from a directory and ingest them into a Neo4j database.
    """
    def __init__(self, file_location, neo4j_manager, debug = False,
                 pdf_reader_url="http://localhost:5010/api/parseDocument?renderFormat=all"):
        """
        Initialize the PDFIngestor.
        
        Args:
            file_location (str): Directory containing PDF files.
            neo4j_manager (Neo4jManager): Instance to handle Neo4j operations.
            pdf_reader_url (str): URL for the PDF parsing API.
        """
        self.file_location = file_location
        self.neo4j_manager = neo4j_manager
        self.debug = debug
        self.pdf_reader = LayoutPDFReader(pdf_reader_url)

    def ingest_all(self):
        """
        Parse all PDF files in the directory and ingest them into Neo4j.
        
        Raises:
            Exception: Propagates errors encountered during ingestion.
        """
        try:
            pdf_files = glob.glob(os.path.join(self.file_location, '*.pdf'))
            if self.debug: print("Found %d PDF files in %s", len(pdf_files), self.file_location)
            logging.info("Found %d PDF files in %s", len(pdf_files), self.file_location)
            start_time = datetime.now()
            for pdf_file in pdf_files:
                if self.debug: print("Processing PDF file: %s", pdf_file)
                logging.info("Processing PDF file: %s", pdf_file)
                try:
                    doc = self.pdf_reader.read_pdf(pdf_file)
                    self.neo4j_manager.ingest_document(doc, pdf_file)
                except Exception as e:
                    if self.debug: print("Failed to process %s: %s", pdf_file, e)
                    logging.error("Failed to process %s: %s", pdf_file, e)
            total_time = datetime.now() - start_time
            if self.debug: print("All PDFs processed in %s", total_time)
            logging.info("All PDFs processed in %s", total_time)
        except Exception as e:
            if self.debug: print("Error during PDF ingestion: %s", e)
            logging.error("Error during PDF ingestion: %s", e)
            raise

class ContextualRetrieval:
    def __init__(self, debug: bool = False, model="gpt-4o-mini"):
        self.debug = debug
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model=model, temperature=0, max_tokens=None, timeout=None, max_retries=2)
        self.graph = Neo4jGraph(refresh_schema=False,url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), 
                          password=os.getenv("NEO4J_PASSWORD"))

    def log(self, level, message):
        if self.debug:
            print(message)
        logger.log(level, message)
    
    def load_single_document(self, file_path: str, is_url: bool = False) -> List[Document]:
        try:
            ext = "url" if is_url else f".{file_path.rsplit('.', 1)[-1]}"
            
            if ext in LOADER_MAPPING:
                loader_class, loader_args = LOADER_MAPPING[ext]
                loader = loader_class(file_path, **loader_args)
                self.log(logging.INFO, f"Loaded document from {file_path}")
                return loader.load()
            
            raise ValueError(f"Unsupported file extension '{ext}'")
        except Exception as e:
            self.log(logging.ERROR, f"Error loading document: {str(e)}")
            return []

    def load_documents(self, source_dir: str, ignored_files: List[str] = []) -> List[Document]:
        all_files = [file for ext in LOADER_MAPPING for file in glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)]
        filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
        results = []
        
        self.log(logging.INFO, f"Found {len(filtered_files)} documents to load.")
        with tqdm(total=len(filtered_files), desc='Loading documents', ncols=80) as pbar:
            for doc in filtered_files:
                results.extend(self.load_single_document(doc))
                pbar.update()
        
        return results

    def process_documents(self, source, ignored_files: List[str] = [], is_url=False, is_dir=False):
        self.log(logging.INFO, f"Processing documents from {source}")
        
        documents = self.load_documents(source, ignored_files) if is_dir else self.load_single_document(source, is_url)
        if not documents:
            self.log(logging.WARNING, "No documents found.")
            return [], []
        
        chunks = self.text_splitter.split_documents(documents)
        contextualized_chunks = self._generate_contextualized_chunks(documents, chunks)
        
        self.log(logging.INFO, "Document processing complete.")
        return chunks, contextualized_chunks
    
    def _generate_contextualized_chunks(self, documents: List[Document], chunks: List[Document]) -> List[Document]:
        return [Document(page_content=f"{self._generate_context(documents, chunk.page_content)}\n\n{chunk.page_content}", metadata=chunk.metadata) for chunk in chunks]

    def _generate_context(self, document: str, chunk: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specializing in document analysis. Provide brief, relevant context for a chunk of text.
        <document>{document}</document>
        <chunk>{chunk}</chunk>
        Context:
        """)
        messages = prompt.format_messages(document=document, chunk=chunk)
        response = self.llm.invoke(messages)
        return response.content
    
    def create_bm25_index(self, chunks: List[Document]):
        return BM25Retriever.from_documents(chunks)
    
    def create_neo4j_index(self, documents, url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), 
                            password=os.getenv("NEO4J_PASSWORD"), index_name="vector", search_type='hybrid'):
        """Creates a Neo4j index from given documents."""      
        try:
            db = Neo4jVector.from_documents(documents, self.embeddings, url=url, username=username, password=password,
                                            index_name=index_name, search_type=search_type)
            self.log(logging.INFO, "Neo4j ingestion complete.")
            return db
        except Exception as e:
            self.log(logging.ERROR, f"Error creating Neo4j index: {e}")
            return None

    def load_neo4j_index(self, url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), 
                          password=os.getenv("NEO4J_PASSWORD"), index_name="vector", search_type='hybrid'):
        """Loads an existing Neo4j index."""
        try:
            index = Neo4jVector.from_existing_index(self.embeddings, url=url, username=username, password=password,
                                                    index_name=index_name, search_type=search_type)
            self.log(logging.INFO, "Loaded Neo4j index successfully.")
            return index
        except Exception as e:
            self.log(logging.ERROR, f"Error loading Neo4j index: {e}")
            return None

    def get_chroma_index(self, index_name="example_collection"):
        """Retrieves a Chroma index by name."""
        try:
            vector_store = Chroma(collection_name=index_name, embedding_function=self.embeddings)
            self.log(logging.INFO, f"Retrieved Chroma index: {index_name}")
            return vector_store
        except Exception as e:
            self.log(logging.ERROR, f"Error retrieving Chroma index: {e}")
            return None

    def add_to_chroma_index(self, document, index_name="example_collection"):
        """Adds a document to an existing Chroma index."""
        try:
            db = self.get_chroma_index(index_name)
            if db is None:
                raise ValueError("Chroma index retrieval failed.")
            db.add_documents(document)
            self.log(logging.INFO, "Document added to Chroma index successfully.")
            return db
        except Exception as e:
            self.log(logging.ERROR, f"Error adding to Chroma index: {e}")
            return None

    def create_graph_data(self, documents, url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), 
                          password=os.getenv("NEO4J_PASSWORD")):
        """Transforms documents into a graph data format."""
        try:
            llm_transformer = LLMGraphTransformer(self.llm)
            graph_document = llm_transformer.convert_to_graph_documents(documents)
            self.log(logging.INFO, "Graph data transformation complete.")
            return graph_document
        except Exception as e:
            self.log(logging.ERROR, f"Error creating graph data: {e}")
            return None

    def store_graph_data(self, graph_documents):
        """Stores graph data in the Neo4j database."""
        try:
            self.graph.add_graph_documents(graph_documents, include_source=True)
            self.log(logging.INFO, "Graph data stored successfully.")
            return self.graph
        except Exception as e:
            self.log(logging.ERROR, f"Error storing graph data: {e}")
            return None

    def get_graph_retriever(self, query):
        """Retrieves graph data using a Cypher-based QA chain."""
        try:
            cypher_chain = GraphCypherQAChain.from_llm(
                cypher_llm=self.llm, qa_llm=self.llm, graph=self.graph, verbose=False
            )
            self.log(logging.INFO, "Graph retriever initialized successfully.")
            return cypher_chain
        except Exception as e:
            self.log(logging.ERROR, f"Error retrieving graph data: {e}")
            return None

    def rerank_retriever(self, retriever):
        compressor = FlashrankRerank()
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
