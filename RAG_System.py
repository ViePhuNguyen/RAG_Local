
from Retrieval_System import RetrievalSystem
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp



class QASystem:
    def __init__(self):
        self.retriever = RetrievalSystem()
        self.llm = self._init_phogpt()
        self.qa_chain = self._create_qa_chain()

    def _init_phogpt(self):
        return LlamaCpp(
            model_path="E:\\RAG_Local_NLP\\Model\\vinallama-7b-chat_q5_0.gguf",
            temperature=0.2,
            max_tokens=1024,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=50,
            n_threads=8,
            verbose=False,
            model_kwargs={
                "main_gpu": 0,
                "tensor_split": [1],
                # "use_mmap": True,
                # "use_mlock": False,
                # "streaming": True,
                "offload_kqv": True,
                "flash_attn": False,
                "rms_norm_eps": 1e-5
            }
        )
    def _create_qa_chain(self):
        """Tạo QA chain với prompt tối ưu cho PhoGPT"""
        prompt_template = """<|system|>
        Bạn là trợ lý AI của Trường Đại học Khoa học và Công nghệ Hà Nội (USTH). 
        Hãy trả lời câu hỏi CHỈ dựa vào thông tin được cung cấp trong phần [Ngữ cảnh].
        Nếu không có thông tin liên quan, hãy trả lời "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu".

        <|user|>
        [Ngữ cảnh]
        {context}

        [Câu hỏi]
        {question}</s>

        <|assistant|>
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def ask(self, question: str) -> dict:
        """Xử lý câu hỏi và trả về kết quả"""
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"].split("[/INST]")[-1].strip(),
                "sources": list(set([doc.metadata["source"] for doc in result["source_documents"]]))
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    # Khởi tạo hệ thống
    qa_system = QASystem()

    # Test câu hỏi
    test_questions = [
        "Chuyên môn của thầy Trần Đình Phong",
    ]

    for q in test_questions:
        print(f"\n=== Câu hỏi: {q} ===")
        response = qa_system.ask(q)

        if "error" in response:
            print(f"Lỗi: {response['error']}")
        else:
            print(f"Trả lời: {response['answer']}")
            print(f"Nguồn tham khảo: {', '.join(response['sources'])}")