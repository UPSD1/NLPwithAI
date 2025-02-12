'''
#installed libraries required
pip install langchain langchain_community tqdm unstructured docx2txt
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade python-docx
pip install sentence-transformers
sudo apt install libreoffice
'''
#================================Import Libraries==================================
import os #provides functions for interacting with the operating system.
import glob #provides functions for file path expansion using wildcards.
from typing import List #List type hint from the typing module
from tqdm import tqdm #provides a progress bar for iterating over sequences.
from langchain.docstore.document import Document #Document class from the langchain.docstore.document module.

# Import document loaders from langchain.document_loaders module for different file types.
from langchain_community.document_loaders import (
    CSVLoader,  # for csv files
    PyMuPDFLoader,  # for pdf files
    TextLoader,  # for txt files
    UnstructuredWordDocumentLoader,  # for doc files
    UnstructuredExcelLoader, #module for xlsx and xls files.
    Docx2txtLoader, #module for docx files.
    UnstructuredURLLoader, #module for URL links
)

from langchain.text_splitter import CharacterTextSplitter #Character splitting

from thirdPartyIntegration import (update_neo4j_vectordb, 
                                   update_pinecone_vectordb,
                                   ingestDocumentNeo4j)

from datetime import datetime
import time
#====================================================================================

#=============================== Variable Declaration ===============================
chunk_size = 1000
chunk_overlap = 50

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xls": (UnstructuredExcelLoader,{}),
    ".xlsx": (UnstructuredExcelLoader,{}),
    "url": (UnstructuredURLLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}

from llmsherpa.readers import LayoutPDFReader
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
file_location = '../data'
#=======================================================================================

#================================ Function Declaration =================================
#load_single_document
def load_single_document(file_path: str, is_url: bool = False) -> List[Document]:
    """
        Loads a single document from a file or URL.

        This function takes a file path or URL and a flag indicating whether it's a URL.
        It then attempts to load the document content based on the file extension or URL
        scheme. Supported file extensions are determined by the `LOADER_MAPPING` global
        variable (assumed to be a dictionary mapping extensions to loader classes and arguments).

        Args:
            file_path (str): The path to the file containing the document or the URL
                of the document.
            is_url (bool, optional): A flag indicating whether `file_path` is a URL.
                Defaults to False (assuming it's a file path).

        Returns:
            List[Document]: A list containing a single document object representing
                the loaded content.

        Raises:
            ValueError: If the file extension is not supported by any available loader.
    """
    if is_url: #if the file path is a url
        ext = "url"
    else:
        ext = "." + file_path.rsplit(".", 1)[-1]  # Extract the file extension from the file path.

    # Check if the file extension is supported by any available loader.
    if ext in LOADER_MAPPING:
        # Get the loader class and loader arguments for the specified file extension.
        loader_class, loader_args = LOADER_MAPPING[ext]

        # Instantiate the loader with the specified file path and loader arguments.
        loader = loader_class(file_path, **loader_args)

        # Load the document using the instantiated loader.
        return loader.load()

    # Raise a ValueError if the file extension is not supported by any available loader.
    raise ValueError(f"Unsupported file extension '{ext}'")

#load documents
def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
        Loads documents from a directory, excluding specified files and handling various file types.

        This function takes a source directory path and an optional list of files to ignore.
        It iterates through supported file extensions (defined in the global `LOADER_MAPPING`)
        and searches the directory recursively for matching files using `glob.glob`. Files
        listed in `ignored_files` are excluded from loading.

        The function uses the `load_single_document` function to process each valid file.
        Loaded documents are appended to a list, and a progress bar (using `tqdm`) is displayed
        to track the loading process.

        Args:
            source_dir (str): The path to the directory containing the documents to load.
            ignored_files (List[str], optional): A list of file paths within the source
                directory to exclude from loading. Defaults to an empty list.

        Returns:
            List[Document]: A list containing the loaded documents represented as document objects.

        Raises:
            OSError: If there is an error accessing the source directory.
    """
    all_files = []  # List to hold all possible files.
    
    # Loop through all acceptable extensions.
    for ext in LOADER_MAPPING:
        # Check if the extension can be found in the directory.
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    # List holding allowable files (files not in the ignored_files list).
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    results = []  # List to store loaded documents.
    # Progress bar to track loading progress.
    with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
        for doc in filtered_files:
            result = load_single_document(doc)  # Load each document using load_single_document function.
            results.extend(result)  # Add loaded document(s) to the results list.
            pbar.update()  # Update progress bar.

    return results

#process documents
def process_documents(source_directory, ignored_files: List[str] = [],is_url = False, is_dir = False, verbose = False) -> List[Document]:
    """
        Processes documents from a directory or URL, handling various input sources and performing text splitting.

        This function takes a source directory path or a URL and optional flags indicating
        the source type (directory or URL). It then performs the following actions:

            1. **Loading:** 
                - If `is_dir` is True, it uses the `load_documents` function to load all
                    supported document types from the directory, excluding files in `ignored_files`.
                - If `is_dir` is False (or unspecified), it treats `source_directory` as a URL or ordinary file
                    and uses the `load_single_document` function to process it (assuming URL support exists).
        
            2. **Filtering:** If the source is a directory and no documents are loaded after
                applying directory checks and exclusions, the function exits with a message.

            3. **Text Splitting:** It initializes a `CharacterTextSplitter` using the specified
                chunk size and overlap configurations (assumed to be defined elsewhere). This
                splitter divides each loaded document into smaller text chunks.

        Args:
            source_directory (str): The path to the directory containing documents or a URL
                pointing to a document.
            ignored_files (List[str], optional): A list of file paths within the source
                directory to exclude from loading (only applicable if `is_dir` is True).
                Defaults to an empty list.
            is_url (bool, optional): A flag indicating whether `source_directory` is a URL.
                If not set, the function attempts to infer the source type. Defaults to False.
            is_dir (bool, optional): A flag explicitly indicating whether `source_directory` is a directory.
                Takes precedence over `is_url` if both are set. Defaults to False.
            verbose (bool, optional): A flag enabling additional informational messages during processing.
                Defaults to False.

        Returns:
            List[Document]: A list containing the processed documents as document objects,
                with each document further split into text chunks.

        Raises:
            OSError: If there is an error accessing the source directory (applicable only
                when `is_dir` is True).
    """
    print(f"Loading documents from {source_directory}")
    
    # Check if source_directory is a directory.
    if is_dir:
        documents = load_documents(source_directory, ignored_files)
        
        # Exit if no documents are loaded.
        if not documents:
            print("No new documents to load")
            exit(0)
            
        print(f"Loaded {len(documents)} new documents from {source_directory}")

    else:
        documents = load_single_document(source_directory, is_url)
        
    # Initialize a text splitter with specified chunk size and overlap.
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Split loaded documents into text chunks using the text splitter.
    documents = text_splitter.split_documents(documents)
    
    # Print information about the splitting process.
    print(f"Split into {len(documents)} chunks of text (max. {chunk_size} tokens each)")
    print("Ingestion complete!")
    return documents

#store documents
def store_document(document: List[Document], db = "neo4j"):
    """
        Stores a list of documents in a knowledge base.

        This function takes a list of documents and a string specifying the target knowledge
        base ("neo4j" or "pinecone"). It attempts to update the chosen knowledge base
        by adding the documents.

        Args:
            document (List[Document]): A list of documents to be stored in the knowledge base.
                Each document should have a text property containing the content to be embedded.
            db (str, optional): The name of the knowledge base to use. Defaults to "neo4j".
                Supported options are "neo4j" and "pinecone".

        Returns:
            Union[Neo4jVectorStore, PineconeVectorStore, Exception]:
                - If successful, returns an instance of either `Neo4jVectorStore` or
                    `PineconeVectorStore` representing the updated knowledge base, depending
                    on the chosen database ("neo4j" or "pinecone").
                - If there is an error, returns the encountered Exception object.

        Raises:
            ValueError: If the provided `db` argument is not a supported knowledge base type.
    """
    if db.lower() ==  "neo4j":
        knowlegde_base = update_neo4j_vectordb(document)
        return knowlegde_base
    
    elif db.lower() == "pinecone":
        knowlegde_base = update_pinecone_vectordb(document)
        return knowlegde_base
    
    return

#load graph with pdf
def load_graph_pdf(file_location = file_location):
    
    pdf_files = glob.glob(file_location + '/*.pdf')
    print(f'#PDF files found: {len(pdf_files)}!')
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)

    # parse documents and create graph
    startTime = datetime.now()

    for pdf_file in pdf_files:
        doc = pdf_reader.read_pdf(pdf_file)

        # find the first / in pdf_file from right
        idx = pdf_file.rfind('/')
        pdf_file_name = pdf_file[idx+1:]

        # open a local file to write the JSON
        with open(pdf_file_name + '.json', 'w') as f:
            # convert doc.json from a list to string
            f.write(str(doc.json))

        ingestDocumentNeo4j(doc, pdf_file)

    print(f'Total time: {datetime.now() - startTime}')

#=========================================================================================