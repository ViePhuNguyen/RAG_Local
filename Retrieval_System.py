
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Optional
import torch


class RetrievalSystem:
    def __init__(self, persist_dir: str = "chroma_db"):
        # Khởi tạo embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = HuggingFaceEmbeddings(
            model_name="bkai-foundation-models/vietnamese-bi-encoder",
            model_kwargs={"device": self.device}
        )

        # Kết nối ChromaDB
        try:
            self.vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embedder
            )
        except Exception as e:
            raise ValueError(f"Failed to load vector store: {str(e)}")

    def retrieve_documents(
            self,
            query: str,
            k: int = 5,
            score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents with options

        Args:
            query: Câu truy vấn
            k: Số lượng documents trả về
            score_threshold: Ngưỡng similarity score (0-1)

        Returns:
            Danh sách documents kèm metadata
        """
        search_kwargs = {"k": k}
        if score_threshold:
            search_kwargs["score_threshold"] = score_threshold

        return self.vector_store.similarity_search(
            query=query,
            **search_kwargs
        )

    def mmr_retrieve(
            self,
            query: str,
            k: int = 5,
            diversity: float = 0.7
    ) -> List[Document]:
        """
        Maximal Marginal Relevance Retrieval

        Args:
            diversity: Mức độ đa dạng (0-1)
                       0 = tương tự thuần túy, 1 = đa dạng tối đa
        """
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            lambda_mult=diversity
        )


# if __name__ == "__main__":
#     # Test retrieval system
#     retriever = RetrievalSystem()
#
#     # Test query
#     test_query = "Điều kiện xin học bổng USTH Ambassador?"
#
#     # Basic retrieval
#     print("=== Similarity Search ===")
#     results = retriever.retrieve_documents(test_query)
#     for doc in results:
#         print(f"Source: {doc.metadata['source']}")
#         print(f"Content: {doc.page_content[:200]}...\n")
#
#     # MMR retrieval
#     print("\n=== MMR Search ===")
#     mmr_results = retriever.mmr_retrieve(test_query)
#     for doc in mmr_results:
#         print(f"Source: {doc.metadata['source']}")
#         print(f"Content: {doc.page_content[:200]}...\n")