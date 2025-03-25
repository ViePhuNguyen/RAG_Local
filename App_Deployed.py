import gradio as gr
from RAG_System import QASystem

# Khởi tạo hệ thống QA
qa_system = QASystem()


def respond(question, history):
    """Xử lý câu hỏi và trả về response định dạng Markdown"""
    try:
        response = qa_system.ask(question)

        if "error" in response:
            return f"⚠️ **Lỗi hệ thống**: {response['error']}"

        answer = f"**Trả lời**: {response['answer']}\n\n"
        answer += "**Nguồn tham khảo**:\n" + "\n".join(
            [f"- {source}" for source in response['sources']]
        )
        return answer

    except Exception as e:
        return f"⛔ **Lỗi nghiêm trọng**: {str(e)}"


# Tạo giao diện
demo = gr.ChatInterface(
    fn=respond,
    title="USTH AI Assistant",
    description="""Hệ thống hỏi đáp thông minh về Trường Đại học Khoa học và Công nghệ Hà Nội""",
    examples=[
        "Điều kiện xét học bổng USTH Ambassador?",
        "Thầy Trần Đình Phong nghiên cứu về lĩnh vực gì?",
        "Hoàng Minh Chi, Nguyễn Hoàng Hà là ai?"
    ],
    theme="soft"
    # Đã bỏ 2 tham số không hỗ trợ
)

if __name__ == "__main__":
    demo.launch(
        server_name="localhost",  # Cho phép truy cập từ mạng
        server_port=7860,  # Port mặc định
        share=False  # Tắt chế độ share public
    )