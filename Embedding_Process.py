import os
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from typing import List
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers.models import Transformer, Pooling, Normalize
from transformers import AutoTokenizer

"""======================= CONFIGURATION PARAMETERS ============================"""
DATA_CONCAT_DIR = "E:\\RAG_Local_NLP\\Data_and_Vector_stored\\Chatbot_data_concat"
MIN_CHUNK_LENGTH = 50  # Minimum character length for valid chunks
CHUNK_SIZE = 400  # https://arxiv.org/abs/2305.15294 256 token với khoảng size 400 cho embedding tiếng việt là size 400
CHUNK_OVERLAP = 80  # Overlap between consecutive chunks

MIN_TOKEN_LENGTH = 20  # Minimum token length for valid chunks
MAX_TOKENS = 256  # Số token tối đa theo khuyến cáo của PhoBERT
TOKEN_OVERLAP = 64  # 25% của MAX_TOKENS

# Khởi tạo tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")


# Hàm tính độ dài theo token
def token_length(text: str) -> int:
    return len(tokenizer.tokenize(text))


"""
==========================RAG Pipeline - 01.Markdown Preprocessing Module=============================

Một trong những cách cải thiện chất lượng của 1 con LLM nhỏ là cải thiện data
vài paper đã chứng minh điều này và tao bắt chúng mày chọn cái Markdown này là vì 
nó nằm trong 1/5 cách chunk data hiệu quả thế nhé

https://arxiv.org/abs/2406.00456 #How chunking effect to the quality of the RAG systems

Chunking là bước tiền xử lý quan trọng trong RAG, ảnh hưởng trực tiếp đến:
Chất lượng retrieval: Kích thước và cách phân đoạn văn bản quyết định độ chính xác của việc tìm kiếm thông tin.
Hiệu suất generator: Độ dài và ngữ nghĩa của chunk tác động đến khả năng tổng hợp câu trả lời của mô hình.

Markdown Splitter : Phương pháp này được thiết kế riêng cho các tài liệu markdown. Nó chia văn bản dựa trên các thành 
phần cụ thể của markdown như tiêu đề, danh sách và khối mã.

====> The reason why using Markdown that usually the administrative documents in the form markdown like the 
PIC_01_Markdonw_form

Ngoài ra nếu chọn chunk vớ vẩn mà sai thì còn ảnh hưởng tới câu trả lời Halluciation có thể tăng lên nhé dkmm
https://arxiv.org/abs/2402.19426
"""

"""======================= MARKDOWN PROCESSING ==================================="""


def preprocess_markdown_files(folder_path: str, headers_to_split_on: list) -> list:
    # Initialize splitters
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False,
        strip_headers=True
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS,
        chunk_overlap=TOKEN_OVERLAP,
        length_function=token_length,
        separators=[
            "\n## ", "\n### ", "\n#### ",  # Ưu tiên split theo heading
            "\n\n", "\n• ", "\n* ",  # Phân tách theo list items
            "\n", ". ", "! ", "? ",  # Phân tách câu
            " ", ""  # Fallback
        ],
        keep_separator=True
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
                    if token_length(sc) > MIN_TOKEN_LENGTH:
                        all_chunks.append({
                            "text": sc,
                            "metadata": {
                                **chunk.metadata,
                                "source": filename,
                                "original_length": token_length(original_content),
                                "cleaned_length": token_length(cleaned_text)
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
    text = re.sub(r'!\[(.*?)]\(.*?\)', r'\1', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Convert markdown links to plain text
    text = re.sub(r'\[(.*?)]\((.*?)\)', r'\1 (\2)', text)

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
        print(chunk['text'][:100] + "...")  # Adjust this to see litter demo
        print("Metadata:", chunk['metadata'])
        print("-" * 50)


"""======================== RAG Pipeline - 02.PhoBERT Embedding Module=====================
Sừ model này là vì nó được train trên tập dữ liệu tiếng việt nearly 20GB Vietnamese documents 
thực chất nó là từ gần 50GB dữ liệu được lọc bỏ những phần bị trùng nhau


Model này base trên Pho-base trên BERT base PIC_02_BERT_arch model Encoder
https://arxiv.org/pdf/2003.00744 Sử dụng nền tảng trên con BERT-base 12 lớp encoder và sử dụng 20GB tiếng việt

Encoder Models
Encoder models phù hợp nhất cho những nhiệm vụ yêu cầu hiểu ngữ nghĩa văn bản và không cần sinh văn bản.
Question Answering (Hỏi đáp)
Mục tiêu: Trả lời câu hỏi dựa trên ngữ cảnh cho trước.

Ví dụ:
Hệ thống RAG dùng encoder để tìm đoạn văn liên quan từ cơ sở dữ liệu, sau đó generator trả lời.
Lý do dùng encoder: Mã hóa cả câu hỏi và ngữ cảnh để tìm thông tin phù hợp.

Mô hình PhoBERT/BERT trả về embeddings cho từng token (từ/cụm từ) trong câu. Ví dụ, câu "Học máy rất thú vị" sẽ có 5 
token embeddings. → Pooling giúp tổng hợp các embeddings này thành 1 vector duy nhất đại diện cho cả câu. Giúp mô 
hình hiểu được ý nghĩa toàn câu thay vì từng từ riêng lẻ. Từ 1 ma trận (số tokens × 768) → 1 vector (768,
). Retrieval: Khi so sánh embedding truy vấn và tài liệu, việc dùng cosine similarity trên các vector đã chuẩn hóa 
giúp tìm kết quả chính xác hơn"""


class PhoBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = "vinai/phobert-base", device: str = "cpu"):
        word_embedding_model = Transformer(model_name, max_seq_length=256)
        pooling_model = Pooling(
            word_embedding_model.get_word_embedding_dimension())  # Gộp thay vì 1 câu có 5 token thì gộp vào lấy tổng
        # của của cả 5 từ đấy để đại diện cho 1 câu
        normalize_model = Normalize()  # Đơn giản nó là chuẩn hóa vector cho dễ tính toán

        self.model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normalize_model],
            device=device
        )
        self.model.max_seq_length = 256  # Set max sequence length cho PhoBERT

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch processing với tối ưu bộ nhớ
        return self.model.encode(
            texts,
            batch_size=32,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True  # Bật normalize cho consistency
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=False
        ).tolist()


def create_chroma_store(chunks):
    texts = [chunk["text"] for chunk in chunks]  # nội dung văn bản chunk
    metadatas = [chunk["metadata"] for chunk in chunks]  # tên văn bản được chunk ra để dễ truy vấn

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
