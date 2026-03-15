"""
Gradio Web Interface for Boston School Chatbot
"""

import os

import gradio as gr

from src.chat import Chatbot


def create_chatbot():
    chatbot = Chatbot()

    def history_to_pairs(history):
        """
        Convert Gradio chat history to (user, assistant) pairs for the model.
        Supports both messages-style and tuple-style histories.
        """
        history = history or []
        if history and isinstance(history[0], dict):
            pairs = []
            pending_user = None
            for item in history:
                role = item.get("role")
                content = item.get("content", "")
                if role == "user":
                    pending_user = content
                elif role == "assistant" and pending_user is not None:
                    pairs.append([pending_user, content])
                    pending_user = None
            return pairs
        return history

    def respond(message, history):
        history = history or []
        model_history = history_to_pairs(history)
        try:
            response = chatbot.get_response(message, model_history)
        except Exception as exc:
            response = (
                "I could not generate a response right now. "
                f"Technical details: {exc}"
            )

        updated_history = model_history + [[message, response]]
        export_path = chatbot.consume_last_export_path()
        downloadable_file = export_path if export_path and os.path.exists(export_path) else None
        return "", updated_history, downloadable_file

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Boston Public School: Support for Parents and Legal Guardians
            Ask me anything about Boston Public Schools enrollment that I will try my best to find what your family needs
            """
        )
        chat_window = gr.Chatbot(label="BPS Enrollment Assistant", height=540, type="tuples")
        message_box = gr.Textbox(
            label="Your message",
            placeholder="Example: My child is entering K2 and we live in 02124",
        )
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")
        export_file = gr.File(
            label="Latest CSV Export",
            interactive=False,
            visible=True,
        )
        gr.Examples(
            examples=[
                "My child is entering 3rd grade and we live in Roxbury. We want Spanish bilingual options.",
                "show more",
                "compare 1 and 3",
                "map options",
                "export csv",
            ],
            inputs=message_box,
        )

        send_btn.click(
            fn=respond,
            inputs=[message_box, chat_window],
            outputs=[message_box, chat_window, export_file],
        )
        message_box.submit(
            fn=respond,
            inputs=[message_box, chat_window],
            outputs=[message_box, chat_window, export_file],
        )
        clear_btn.click(
            fn=lambda: (chatbot.reset_state() or "", [], None),
            outputs=[message_box, chat_window, export_file],
        )

    return demo


if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
