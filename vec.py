import os
import tensorflow as tf
import warnings
from sentence_transformers import SentenceTransformer
import singlestoredb as s2
import pandas as pd  # Import pandas for CSV handling

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warning messages

# Suppress specific TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to create a new connection
def create_connection():
    return s2.connect('<Login Details>')

# Load the BGE-M3 model using Sentence Transformers
model = SentenceTransformer('BAAI/bge-m3')

# Function to read sentences from a CSV file and concatenate columns
def read_sentences_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')  # Attempt to read with UTF-8 encoding
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Fallback to ISO-8859-1 encoding
    # Concatenate the relevant columns into a single string for each row
    return df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()  # Adjust as needed

# Specify the path to your CSV file
file_path = 'tr2.csv'  # Update this to your actual file path

# Read sentences from the CSV
sentences = read_sentences_from_csv(file_path)

# Generate embeddings for the sentences
embeddings = model.encode(sentences, show_progress_bar=True)

# Insert embeddings into SingleStoreDB
try:
    with create_connection() as conn:  # Create a new connection for this block
        with conn.cursor() as cur:
            # Create a table to store embeddings if it doesn't exist
            cur.execute(""" 
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    content TEXT,
                    embedding BLOB
                )
            """)

            # Insert each concatenated text and its corresponding embedding into the table
            for sentence, embedding in zip(sentences, embeddings):
                cur.execute(""" 
                    INSERT INTO embeddings (content, embedding) VALUES (%s, %s)
                """, (sentence, embedding.tobytes()))  # Convert numpy array to bytes

            # Commit the transaction
            conn.commit()
            print("Insertion successful.")
except Exception as e:
    print(f"Insertion failed: {e}")

# Performing a similarity search can be done here (not shown in this example)
