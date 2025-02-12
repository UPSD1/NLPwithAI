import os
import glob
import logging
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import (
    CSVLoader, PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, Docx2txtLoader, UnstructuredURLLoader
)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_SECRET_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

class ContextualRetrieval:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2)
        
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
    
    def create_neo4j_index(self, documents, url, username, password, index_name="vector"):
        db = Neo4jVector.from_documents(documents, self.embeddings, url=url, username=username, password=password, index_name=index_name)
        self.log(logging.INFO, "Neo4j ingestion complete.")
        return db
    
    def load_neo4j_index(self, url, username, password, index_name="vector", search_type='hybrid'):
        return Neo4jVector.from_existing_index(self.embeddings, url=url, username=username, password=password, index_name=index_name, search_type=search_type)
    
    def rerank_retriever(self, retriever):
        compressor = FlashrankRerank()
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


# LOADER_MAPPING = {
#     ".csv": (CSVLoader, {}),
#     ".doc": (UnstructuredWordDocumentLoader, {}),
#     ".docx": (Docx2txtLoader, {}),
#     ".pdf": (PyMuPDFLoader, {}),
#     ".txt": (TextLoader, {"encoding": "utf8"}),
#     ".xls": (UnstructuredExcelLoader,{}),
#     ".xlsx": (UnstructuredExcelLoader,{}),
#     "url": (UnstructuredURLLoader, {}),
#     # Add more mappings for other file extensions and loaders as needed
# }

# class ContextualRetrieval:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100,
#         )
#         self.debug = False
#         self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#         self.llm = ChatOpenAI(model="gpt-4o",temperature=0,
#                               max_tokens=None,timeout=None,
#                               max_retries=2,)
        
#     #load_single_document
#     def load_single_document(self, file_path: str, is_url: bool = False) -> List[Document]:
#         """
#             Loads a single document from a file or URL.

#             This function takes a file path or URL and a flag indicating whether it's a URL.
#             It then attempts to load the document content based on the file extension or URL
#             scheme. Supported file extensions are determined by the `LOADER_MAPPING` global
#             variable (assumed to be a dictionary mapping extensions to loader classes and arguments).

#             Args:
#                 file_path (str): The path to the file containing the document or the URL
#                     of the document.
#                 is_url (bool, optional): A flag indicating whether `file_path` is a URL.
#                     Defaults to False (assuming it's a file path).

#             Returns:
#                 List[Document]: A list containing a single document object representing
#                     the loaded content.

#             Raises:
#                 ValueError: If the file extension is not supported by any available loader.
#         """
#         if is_url: #if the file path is a url
#             ext = "url"
#         else:
#             ext = "." + file_path.rsplit(".", 1)[-1]  # Extract the file extension from the file path.

#         # Check if the file extension is supported by any available loader.
#         if ext in LOADER_MAPPING:
#             # Get the loader class and loader arguments for the specified file extension.
#             loader_class, loader_args = LOADER_MAPPING[ext]

#             # Instantiate the loader with the specified file path and loader arguments.
#             loader = loader_class(file_path, **loader_args)

#             # Load the document using the instantiated loader.
#             return loader.load()

#         # Raise a ValueError if the file extension is not supported by any available loader.
#         raise ValueError(f"Unsupported file extension '{ext}'")

#     #load documents
#     def load_documents(self, source_dir: str, ignored_files: List[str] = []) -> List[Document]:
#         """
#             Loads documents from a directory, excluding specified files and handling various file types.

#             This function takes a source directory path and an optional list of files to ignore.
#             It iterates through supported file extensions (defined in the global `LOADER_MAPPING`)
#             and searches the directory recursively for matching files using `glob.glob`. Files
#             listed in `ignored_files` are excluded from loading.

#             The function uses the `load_single_document` function to process each valid file.
#             Loaded documents are appended to a list, and a progress bar (using `tqdm`) is displayed
#             to track the loading process.

#             Args:
#                 source_dir (str): The path to the directory containing the documents to load.
#                 ignored_files (List[str], optional): A list of file paths within the source
#                     directory to exclude from loading. Defaults to an empty list.

#             Returns:
#                 List[Document]: A list containing the loaded documents represented as document objects.

#             Raises:
#                 OSError: If there is an error accessing the source directory.
#         """
#         all_files = []  # List to hold all possible files.
        
#         # Loop through all acceptable extensions.
#         for ext in LOADER_MAPPING:
#             # Check if the extension can be found in the directory.
#             all_files.extend(
#                 glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
#             )
#         # List holding allowable files (files not in the ignored_files list).
#         filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
#         results = []  # List to store loaded documents.
#         # Progress bar to track loading progress.
#         with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
#             for doc in filtered_files:
#                 result = self.load_single_document(doc)  # Load each document using load_single_document function.
#                 results.extend(result)  # Add loaded document(s) to the results list.
#                 pbar.update()  # Update progress bar.

#         return results

#     #process documents
#     def process_documents(self, source_directory, ignored_files: List[str] = [],is_url = False, is_dir = False) -> List[Document]:
#         """
#             Processes documents from a directory or URL, handling various input sources and performing text splitting.

#             This function takes a source directory path or a URL and optional flags indicating
#             the source type (directory or URL). It then performs the following actions:

#                 1. **Loading:** 
#                     - If `is_dir` is True, it uses the `load_documents` function to load all
#                         supported document types from the directory, excluding files in `ignored_files`.
#                     - If `is_dir` is False (or unspecified), it treats `source_directory` as a URL or ordinary file
#                         and uses the `load_single_document` function to process it (assuming URL support exists).
            
#                 2. **Filtering:** If the source is a directory and no documents are loaded after
#                     applying directory checks and exclusions, the function exits with a message.

#                 3. **Text Splitting:** It initializes a `CharacterTextSplitter` using the specified
#                     chunk size and overlap configurations (assumed to be defined elsewhere). This
#                     splitter divides each loaded document into smaller text chunks.

#             Args:
#                 source_directory (str): The path to the directory containing documents or a URL
#                     pointing to a document.
#                 ignored_files (List[str], optional): A list of file paths within the source
#                     directory to exclude from loading (only applicable if `is_dir` is True).
#                     Defaults to an empty list.
#                 is_url (bool, optional): A flag indicating whether `source_directory` is a URL.
#                     If not set, the function attempts to infer the source type. Defaults to False.
#                 is_dir (bool, optional): A flag explicitly indicating whether `source_directory` is a directory.
#                     Takes precedence over `is_url` if both are set. Defaults to False.
#                 verbose (bool, optional): A flag enabling additional informational messages during processing.
#                     Defaults to False.

#             Returns:
#                 List[Document]: A list containing the processed documents as document objects,
#                     with each document further split into text chunks.

#             Raises:
#                 OSError: If there is an error accessing the source directory (applicable only
#                     when `is_dir` is True).
#         """
#         if self.debug: print(f"Loading documents from {source_directory}")
        
#         # Check if source_directory is a directory.
#         if is_dir:
#             documents = load_documents(source_directory, ignored_files)
            
#             # Exit if no documents are loaded.
#             if not documents:
#                 if self.debug: print("No new documents to load")
#                 exit(0)
                
#             if self.debug: print(f"Loaded {len(documents)} new documents from {source_directory}")

#         else:
#             documents = load_single_document(source_directory, is_url)
            
#         # Split loaded documents into text chunks using the text splitter.
#         chunks = self.text_splitter.split_documents([documents])
#         contextualized_chunks = self._generate_contextualized_chunks([documents], chunks)
        
#         # Print information about the splitting process.
#         if self.debug: print("Ingestion complete!")
#         return chunks, contextualized_chunks
        
#     def _generate_contextualized_chunks(self, document: str, chunks: List[Document]) -> List[Document]:
#         contextualized_chunks = []
#         for chunk in chunks:
#             context = self._generate_context(document, chunk.page_content)
#             contextualized_content = f"{context}\n\n{chunk.page_content}"
#             contextualized_chunks.append(Document(page_content=contextualized_content, metadata=chunk.metadata))
#         return contextualized_chunks
    
#     def _generate_context(self, document: str, chunk: str) -> str:
#         prompt = ChatPromptTemplate.from_template("""
#         You are an AI assistant specializing in document analysis. Your task is to provide brief, relevant context for a chunk of text from the given document.
#         Here is the document:
#         <document>
#         {document}
#         </document>

#         Here is the chunk we want to situate within the whole document:
#         <chunk>
#         {chunk}
#         </chunk>

#         Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
#         1. Identify the main topic or concept discussed in the chunk.
#         2. Mention any relevant information or comparisons from the broader document context.
#         3. If applicable, note how this information relates to the overall theme or purpose of the document.
#         4. Include any key figures, dates, or percentages that provide important context.
#         5. Do not use phrases like "This chunk discusses" or "This section provides". Instead, directly state the context.

#         Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.

#         Context:
#         """)
#         messages = prompt.format_messages(document=document, chunk=chunk)
#         response = self.llm.invoke(messages)
#         return response.content
    
#     def create_bm25_index(self, chunks: List[Document]) -> List[Document] :
#         retriever = BM25Retriever.from_documents(chunks)
#         return retriever
    
#     def create_neo4j_index(self, document, url, username, 
#                             password, index_name="vector")-> List[Document]:
#         db = Neo4jVector.from_documents(
#             document, self.embeddings, url=url, username=username, 
#             password=password, index_name=index_name
#         )
#         if self.debug: print("Neo4j ingestion complete")
#         return db
    
#     def load_neo4j_index(self, document, url, username, 
#                             password, search_type = 'hybrid',index_name="vector"):
#         index =  Neo4jVector.from_existing_index(
#             self.embeddings,url=url,username=username,password=password,
#             index_name=index_name, search_type=search_type)
#         return index
    
#     def rerank_retriever(self, retriever):
#         compressor = FlashrankRerank()
#         compression_retriever = ContextualCompressionRetriever(
#             base_compressor=compressor, base_retriever=retriever
#         )
#         return compression_retriever

    
    

# references
# https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/
# https://python.langchain.com/docs/integrations/retrievers/bm25/
# https://medium.com/@manoranjan.rajguru/building-a-contextual-retrieval-system-for-improving-rag-accuracy-1d9344a604e7
# https://neo4j.com/labs/genai-ecosystem/langchain/

