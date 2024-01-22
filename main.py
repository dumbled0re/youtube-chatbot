import json
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
                    "description": "質問事項 e.g. 動画内容を要約して",
                    # "description": "質問事項",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_keyword",
        "description": "キーワードを抽出する",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "キーワード e.g. 〇〇について、〇〇に関して",
                },
            },
            "required": ["keyword"],
        }
    },
    {
        "name": "get_index",
        "description": "インデックス番号を抽出する",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "description": "インデックス番号 e.g. 1",
                },
            },
            "required": ["index"],
        }
    },
]


def get_question(llm, message):
    chat_completion = llm.predict_messages(
        [HumanMessage(content=message)],
        functions=functions,
    )

    return chat_completion
    # return message.additional_kwargs


def get_keyword(llm, message):
    chat_completion = llm.predict_messages(
        [HumanMessage(content=message)],
        functions=functions,
    )

    return chat_completion


def get_index(llm, message):
    chat_completion = llm.predict_messages(
        [HumanMessage(content=message)],
        functions=functions,
    )

    return chat_completion


def main():
    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"),
                     model_name="gpt-3.5-turbo-16k",
                     temperature=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # chunkの文字数
        chunk_overlap=10,  # chunk間の重複文字数
    )

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
            chunk_document = text_splitter.split_text(document[0].page_content)
            # TODO: ここでチャンク数分の番号と文章のディクショナリを作成したらいいかも
            # TODO: それをうまいことgptに入れて、質問に答えられるようにする
            num_list = [i for i in range(len(chunk_document))]
            chunk_dict = dict(zip(num_list, chunk_document))
            print(f"chunk_dict: {chunk_dict}")
            video_content = document[0].page_content
            # print(f"document: {document}\n")
            # print(f"chunk_document: {chunk_document}\n")
            # print(f"video_content: {video_content}")

    if user_input := st.chat_input("テキスト入力"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("生成中..."):
            # response = llm(st.session_state.messages)
            response = get_question(llm, user_input)
            print(f"###### response.additional_kwargs: {response.additional_kwargs}")
            if response.additional_kwargs:
                if response.additional_kwargs["function_call"]["name"] == "get_question":
                    print(f"response: {response}")
                    print(
                        f"response.additional_kwargs: {response.additional_kwargs}"
                        )

                    question = json.loads(
                        response.additional_kwargs["function_call"]["arguments"]
                        ).get("question")
                    print(f"question: {question}")

                    # system_template = "あなたは、質問者からの質問を{language}で回答するAIです。"
                    system_template = "あなたは、質問者からの質問を回答するAIです。"
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
                elif response.additional_kwargs["function_call"]["name"] == "get_keyword":
                    print("get_keywordが呼び出されました。")
                    keyword = json.loads(
                        response.additional_kwargs["function_call"]["arguments"]
                        ).get("keyword")
                    print(f"keyword: {keyword}")

                    system_template = "あなたは、リストの中にあるチャンクを読み取ってインデックス番号を回答するAIです。"
                    # system_template = """
                    #     あなたは、Pythonのディクショナリ内にあるキーがインデックス番号とテキストを読み取って適切なインデックス番号を回答するAIです。
                    # """
                    human_template = """
                        以下のディクショナリ内にあるインデックス番号とテキストを元に{keyword}の説明がされてあるインデックス番号は何番でしょうか。

                        {chunk_dict}

                        回答形式は「{keyword}の説明は{index}番です。」としてください。
                        もしも、{keyword}の説明がない場合は「{keyword}の説明はありません。」としてください。
                    """

                    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
                    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

                    print(f"system_message_prompt: {system_message_prompt}")
                    print(f"human_message_prompt: {human_message_prompt}")

                    chat_prompt = ChatPromptTemplate.from_messages(
                        [system_message_prompt, human_message_prompt])
                    prompt_message_list = chat_prompt.format_prompt(
                        keyword=keyword,
                        chunk_dict=chunk_dict,
                        index="インデックス").to_messages()
                    second_response = llm(prompt_message_list)
                    print(f"second_response: {second_response}")
                    print(f"type(second_response.content): {type(second_response.content)}")
                    third_response = get_index(llm, second_response.content)
                    if third_response.additional_kwargs["function_call"]["name"] == "get_index":
                        index = json.loads(third_response.additional_kwargs["function_call"]["arguments"]).get("index")
                        print(f"index: {index}")
                        ai_answer = f"{keyword}の説明は{int(index)*1}分くらいからです。"
                        st.session_state.messages.append(
                            AIMessage(content=ai_answer))
                    else:
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
