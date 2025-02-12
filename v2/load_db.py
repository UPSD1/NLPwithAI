import logging
import os
from dotenv import load_dotenv
from db_helper import Neo4jManager, PDFIngestor, ContextualRetrieval

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

class DocumentProcessor:
    """
    Handles the processing and indexing of documents from a given directory.
    Uses logging for monitoring and debugging.
    """
    def __init__(self, pdf_dir: str, debug: bool = False):
        """
        Initializes the document processor.
        
        :param pdf_dir: Directory containing PDF documents.
        :param debug: If True, enables print statements along with logs.
        """
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_SECRET_KEY')
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_username = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        self.pdf_dir = pdf_dir
        self.debug = debug
        self.cr = ContextualRetrieval(debug=debug)

        # Configure logging
        self.logger = logging.getLogger(__name__)

    def log(self, level, message):
        """
        Logs a message at the given level and prints it if debug mode is enabled.
        
        :param level: Logging level (DEBUG, INFO, ERROR, etc.).
        :param message: The message to log.
        """
        if self.debug:
            print(message)
        self.logger.log(level, message)

    def get_files(self):
        """
        Retrieves all files in the specified directory.
        
        :return: List of file paths.
        """
        try:
            files = [os.path.join(self.pdf_dir, f) for f in os.listdir(self.pdf_dir) if os.path.isfile(os.path.join(self.pdf_dir, f))]
            self.log(logging.INFO, f"Found {len(files)} files in directory: {self.pdf_dir}")
            return files
        except Exception as e:
            self.log(logging.ERROR, f"Error accessing directory {self.pdf_dir}: {str(e)}")
            return []

    def process_documents(self):
        """
        Processes each document one by one from the directory.
        Extracts chunks and contextualized chunks.
        
        :return: Tuple containing lists of chunks and contextualized chunks.
        """
        files = self.get_files()
        chunk_list = []
        contextualized_chunk_list = []

        for file_path in files:
            self.log(logging.INFO, f"Processing file: {file_path}")
            try:
                chunks, contextualized_chunks = self.cr.process_documents(file_path, is_dir=False)
                chunk_list.extend(chunks)
                contextualized_chunk_list.extend(contextualized_chunks)
                self.log(logging.INFO, f"Successfully processed: {file_path}")
            except Exception as e:
                self.log(logging.ERROR, f"Error processing {file_path}: {str(e)}")

        return chunk_list, contextualized_chunk_list

    def create_indexes(self, chunk_list, contextualized_chunk_list):
        """
        Creates Neo4j indexes for normal and contextualized document chunks.
        """
        try:
            self.log(logging.INFO, "Creating Neo4j indexes...")
            
            # Index for contextualized chunks
            contextualized_neo4j_index = self.cr.create_neo4j_index(
                contextualized_chunk_list,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name="lectureNoteContextualIndex",
                search_type='hybrid'
            )
            self.log(logging.INFO, "Contextualized Neo4j index created successfully.")

            # Index for normal chunks
            normal_neo4j_index = self.cr.create_neo4j_index(
                chunk_list,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name="lectureNoteIndex",
                search_type='hybrid'
            )
            self.log(logging.INFO, "Normal Neo4j index created successfully.")

        except Exception as e:
            self.log(logging.ERROR, f"Error creating Neo4j indexes: {str(e)}")

    def run(self):
        """
        Runs the document processing and indexing workflow.
        """
        self.log(logging.INFO, "Starting document processing...")
        chunk_list, contextualized_chunk_list = self.process_documents()
        self.create_indexes(chunk_list, contextualized_chunk_list)
        self.log(logging.INFO, "Document processing and indexing complete.")


# # Instantiate and run the processor
# processor = DocumentProcessor(pdf_dir="/content/data", debug=True)
# processor.run()

pdfingestor = PDFIngestor("/Data", Neo4jManager, debug=False)
pdfingestor.ingest_all()