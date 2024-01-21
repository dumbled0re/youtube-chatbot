import json
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

functions = [
    {
        "name": "get_question",
        "description": "質問を抽出する",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "質問事項 e.g. 要約",
                    # "description": "質問事項",
                },
            },
            "required": ["question"],
        },
    }
]


def get_question(llm, message):
    chat_completion = llm.predict_messages(
        [HumanMessage(content=message)],
        functions=functions,
    )

    return chat_completion
    # return message.additional_kwargs


def generate_responses_from_video_transcription(llm, message):
    pass


def main():
    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"),
                     model_name="gpt-3.5-turbo-16k",
                     temperature=0)

    # ページ上部の部分
    st.set_page_config(
        page_title="Youtube chatbot",
        page_icon="🤗"
    )
    st.header("Youtube chatbot 🤗")

    # チャット履歴の初期化
    if not st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # ユーザーの入力を監視
    # if url := st.text_input("Youtube URL: ", key="input"):
    url = st.text_input("Youtube URL: ", key="input")
    print(f"url: {url}")
    if url:
        with st.spinner("Fetching Content ..."):
            loader = YoutubeLoader.from_youtube_url(
                url,
                # add_video_info=True,  # タイトルや再生数も取得できる
                language=['ja']
            )
            document = loader.load()
            video_content = document[0].page_content
            print(f"document: {document}\n")
            # print(f"video_content: {video_content}")

    if user_input := st.chat_input("テキスト入力"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("生成中..."):
            # response = llm(st.session_state.messages)
            response = get_question(llm, user_input)
            if response.additional_kwargs:
                print(f"response: {response}")
                print(f"response.additional_kwargs: {response.additional_kwargs}")
                question = json.loads(
                    response.additional_kwargs["function_call"]["arguments"]).get("question")
                print(f"question: {question}")

                system_template = "あなたは、質問者からの質問を{language}で回答するAIです。"
                human_template = """
                    以下のテキストを元に「{question}」についての質問に答えてください。

                    {document}
                """

                system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
                human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

                print(f"system_message_prompt: {system_message_prompt}")
                print(f"human_message_prompt: {human_message_prompt}")

                chat_prompt = ChatPromptTemplate.from_messages(
                    [system_message_prompt, human_message_prompt])
                prompt_message_list = chat_prompt.format_prompt(
                    language="日本語",
                    question=question,
                    document=video_content).to_messages()
                second_response = llm(prompt_message_list)
                print(f"second_response: {second_response}")

                st.session_state.messages.append(
                    AIMessage(content=second_response.content))
            else:
                print("function callingが呼び出されませんでした。")
                response = llm(st.session_state.messages)
                st.session_state.messages.append(
                    AIMessage(content=response.content))

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    print(f"messages: {messages}")
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)


if __name__ == '__main__':
    main()
