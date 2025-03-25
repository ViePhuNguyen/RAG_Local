import os
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers.models import Transformer, Pooling, Normalize
"""
==========================ĐÔI LỜI HOÀNG DŨNG=============================

RAG Pipeline - 01.Markdown Preprocessing Module

Một trong những cách cải thiện chất lượng của 1 con LLM nhỏ là cải thiện data
vài paper đã chứng minh điều này và tao bắt chúng mày chọn cái Markdown này là vì 
nó nằm trong 1/8 cách chunk data hiệu quả thế nhé

"""

"""======================= CONFIGURATION PARAMETERS ============================"""
DATA_CONCAT_DIR = "E:\\RAG_Local_NLP\\Data_and_Vector_stored\\Chatbot_data_concat"
MIN_CHUNK_LENGTH = 50  # Minimum character length for valid chunks
CHUNK_SIZE = 400  # Target size for text chunks
CHUNK_OVERLAP = 80  # Overlap between consecutive chunks


"""======================= MARKDOWN PROCESSING ==================================="""
def preprocess_markdown_files(folder_path: str, headers_to_split_on: list) -> list:

    # Initialize splitters
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    all_chunks = []
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    print(f"Found {len(txt_files)} markdown files in {folder_path}")

    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            if not md_content.strip():
                print("Empty file, skipping")
                continue

            # Stage 1: Split by markdown headers
            header_chunks = header_splitter.split_text(md_content)
            print(f"Stage 1: Created {len(header_chunks)} header-based chunks")

            # Process each header chunk
            for chunk in header_chunks:
                original_content = chunk.page_content
                cleaned_text = clean_markdown(original_content)

                # Skip empty content after cleaning
                if not cleaned_text.strip():
                    continue

                # Stage 2: Split into size-optimized chunks
                sub_chunks = text_splitter.split_text(cleaned_text)

                # Add metadata and store valid chunks
                for sc in sub_chunks:
                    if len(sc) > MIN_CHUNK_LENGTH:
                        all_chunks.append({
                            "text": sc,
                            "metadata": {
                                **chunk.metadata,
                                "source": filename,
                                "original_length": len(original_content),
                                "cleaned_length": len(cleaned_text)
                            }
                        })

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    # Print processing summary
    print(f"\nProcessed {len(all_chunks)} chunks from {len(txt_files)} files")
    print_sample_chunks(all_chunks)

    return all_chunks

"""========================= SPECIAL CHARACTERS HANDLE ========================="""
def clean_markdown(text: str) -> str:

    # Preserve code block content
    text = re.sub(r'```.*?\n(.*?)```', r'\1', text, flags=re.DOTALL)

    # Preserve inline code
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Extract image alt text
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Convert markdown links to plain text
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text)

    # Remove special markdown characters
    text = re.sub(r'[#*\-_>|]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

"""======================== DEMO FOR DEBUG PURPOSE =============================="""
def print_sample_chunks(chunks: list, sample_size: int = 3) -> None:
    print("\nSample chunks:")
    for idx, chunk in enumerate(chunks[:sample_size]):
        print(f"\nChunk {idx + 1} ({len(chunk['text'])} chars):")
        print(chunk['text'][:100] + "...") # Adjust this to see litter demo
        print("Metadata:", chunk['metadata'])
        print("-" * 50)


"""========================= EMBEDDING BLOCK======================"""
# class VietnameseEmbeddings(HuggingFaceEmbeddings):
#     def __init__(self, model_name: str, device: str = "cpu"):
#         super().__init__(
#             model_name=model_name,
#             model_kwargs={"device": device}
#         )
#         # Initialize SentenceTransformer here, not as a direct attribute
#         self._model = SentenceTransformer(model_name, device=device, cache_folder="models")
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self._model.encode(
#             texts,
#             convert_to_tensor=False,
#             show_progress_bar=True,
#             normalize_embeddings=True
#         ).tolist()
#
#     def embed_query(self, text: str) -> List[float]:
#         return self._model.encode(
#             text,
#             convert_to_tensor=False,
#             normalize_embeddings=True
#         ).tolist()
"""======================== PHOBERT EMBEDDING BLOCK====================="""
class PhoBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = "vinai/phobert-base", device: str = "cpu"):
        # Khởi tạo các thành phần của Sentence Transformer
        word_embedding_model = Transformer(model_name, max_seq_length=256)
        pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
        normalize_model = Normalize()

        self.model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normalize_model],
            device=device
        )
        self.model.max_seq_length = 256  # Set max sequence length cho PhoBERT

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=False
        ).tolist()
def create_chroma_store(chunks):
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Xác định device
    device = "cuda" if torch.cuda.is_available() else "cpu"



    # Khởi tạo embedding
    embedder = PhoBERTEmbeddings(
        model_name="vinai/phobert-base",
        device=device
    )

    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        persist_directory="Data_Base_Vector",
        collection_metadata={
            "hnsw:space": "cosine",
                "phobert_max_seq_length": 256,
                "phobert_pooling_mode": "mean",
                "phobert_do_lower_case": False  # Giữ nguyên chữ hoa/chữ thường cho tiếng Việt

        },
    )

    print(f"Vector store created with {len(texts)} chunks")
    return vector_db



if __name__ == "__main__":

    """================MARKDOWN DOCUMENT PRE-PROCESS======================="""
    processed_chunks = preprocess_markdown_files(
        folder_path=DATA_CONCAT_DIR,
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
    )
    """================WORD EMBEDDING PROCESS======================="""
    chroma_db = create_chroma_store(processed_chunks)
    print("ChromaDB created with", chroma_db._collection.count(), "documents")