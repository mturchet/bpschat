"""
Gradio Web Interface for Boston School Chatbot
"""

import os

import gradio as gr

from src.chat import Chatbot, GREETING_TEMPLATE


def create_chatbot():
    chatbot = Chatbot()

    # Initial greeting shown when page loads
    initial_history = [["", GREETING_TEMPLATE]]

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
        # Filter out the initial empty-user greeting from model history
        model_history = [[u, b] for u, b in model_history if u]
        try:
            response = chatbot.get_response(message, model_history)
        except Exception as exc:
            response = (
                "I could not generate a response right now. "
                f"Technical details: {exc}"
            )

        updated_history = history + [[message, response]]
        export_path = chatbot.consume_last_export_path()
        downloadable_file = export_path if export_path and os.path.exists(export_path) else None
        return "", updated_history, downloadable_file

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Boston Public Schools Enrollment Assistant
            *Helping Boston families find the right school — powered by real BPS eligibility data.*
            """
        )
        chat_window = gr.Chatbot(
            label="BPS Enrollment Assistant",
            height=540,
            type="tuples",
            value=initial_history,
        )
        message_box = gr.Textbox(
            label="Your message",
            placeholder="Example: My child is entering K2 and we live at 100 Warren St, Boston 02119",
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
                "My child is entering 3rd grade and we live at 100 Warren St, Boston 02119. We speak English at home.",
                "What is a Section 504 plan?",
                "show more",
                "compare 1 and 3",
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
            fn=lambda: (chatbot.reset_state() or "", initial_history, None),
            outputs=[message_box, chat_window, export_file],
        )

    return demo


if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
