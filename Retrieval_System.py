from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Optional, Union, Tuple
import torch
import os
from Embedding_Process import PhoBERTEmbeddings


class RetrievalSystem:
    def __init__(self, persist_dir: str = "Data_Base_Vector"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedder = PhoBERTEmbeddings(device=self.device)
        try:
            self.vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embedder,
            )
        except Exception as e:
            raise RuntimeError(f"Không thể kết nối ChromaDB: {str(e)}")

    def retrieve_documents(
            self,
            query: str,
            k: int = 5,
            score_threshold: Optional[float] = None
    ) -> list[tuple[Document, float]]:
        search_kwargs = {"k": min(k, 20)}
        if score_threshold:
            search_kwargs["score_threshold"] = max(0.0, min(1.0, score_threshold))

        return self.vector_store.similarity_search_with_score(
            query=query,
            **search_kwargs
        )

    def mmr_retrieve(
            self,
            query: str,
            k: int = 5,
            diversity: float = 0.7
    ) -> List[Document]:
        diversity = max(0.0, min(1.0, diversity))
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=min(k, 20),
            lambda_mult=diversity
        )


if __name__ == "__main__":
    try:
        retriever = RetrievalSystem()
        query = "Chuyên môn của cô Nghiêm Thị Phương (khoa ICT) là gì?"
        results = retriever.retrieve_documents(query)

        for doc, score in results:
            print(f"\nScore: {score:.3f}")
            print(f"Source: {doc.metadata['source']}")
            print(f"Content: {doc.page_content[:200]}...")

    except Exception as e:
        print(f"Error: {str(e)}")
