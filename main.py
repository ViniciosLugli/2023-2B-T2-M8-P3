import gradio as gr
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Chatbot:
    MAIN_PROMPT = "Você é um ajudante de almoxarifado e deve ajudar com o que for necessário."

    def __init__(self):
        self.llm = OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=512)

    def generate_chat_history(self, history):
        parsed_history = "CHAT HISTORY:\n"
        for i, (user, bot) in enumerate(history):
            parsed_history += f"USER: {user}\nBOT: {bot}\n"
        parsed_history += "\nEND OF HISTORY\n"
        return parsed_history

    def chatbot_response(self, message, history):
        parsed_history = self.generate_chat_history(history)
        prompt = f"{Chatbot.MAIN_PROMPT}\n{parsed_history}\n{message}"

        response = ""
        for partial_message in self.llm.stream(prompt):
            response += partial_message
            yield response


class ChatInterface:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def launch(self):
        chat_interface = gr.ChatInterface(
            self.chatbot.chatbot_response,
            examples=[
                "Quais EPIs são necessários para operar um torno mecânico?",
                "Quais EPIs são necessários para operar uma fresadora?",
                "O que fazer em caso de incêndio?",
                "O que fazer em caso de vazamento de produtos químicos?",
            ],
            title="Chatbot",
        )
        chat_interface.launch()


if __name__ == "__main__":
    chatbot_instance = Chatbot()
    chat_interface_instance = ChatInterface(chatbot_instance)
    chat_interface_instance.launch()
