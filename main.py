import json
import math
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

functions = [
    {
        "name": "generate_video_response",
        "description": "質問を抽出する",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "質問事項 e.g. 動画内容を要約して",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "generate_video_time_response",
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
                    "description": "インデックス番号 e.g. 1番",
                },
            },
            "required": ["index"],
        }
    },
]


def get_question(llm, question):
    messages = llm.predict_messages(
        [HumanMessage(content=question)],
        functions=functions,
    )

    return messages.additional_kwargs

# TODO: 以下のget_keywordはいらないのでは？
def get_keyword(llm, message):
    messages = llm.predict_messages(
        [HumanMessage(content=message)],
        functions=functions,
    )

    return messages.additional_kwargs


def get_index(llm, message):
    messages = llm.predict_messages(
        [HumanMessage(content=message)],
        functions=functions,
    )

    return messages.additional_kwargs


def set_up_page() -> None:
    """
    ページの設定とヘッダーを表示する関数
    """

    # ページ上部の部分の設定
    st.set_page_config(
        page_title="Youtube Chatbot",
        page_icon="🤖",
    )

    # ヘッダーの表示
    st.header("Youtube Chatbot 🤖")


def init_session_state(session_state: dict) -> dict[str, list[str]]:
    """セッションステートを初期化する関数

    Args:
        session_state (dict): セッションステート
    Returns:
        dict: 初期化されたセッションステート
    """

    if not session_state:
        session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    return session_state


def display_chat_history(messages: list) -> None:
    """チャット履歴を表示する関数

    Args:
        messages (list): チャットメッセージのリスト
    """

    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)


def convert_seconds(seconds: int) -> str:
    """秒を分や時間に換算する

    Args:
        seconds (int): 換算対象の秒数
    Returns:
        str: 換算結果の数値または"時間:分"の文字列
    """

    minutes = 0
    hours = 0

    if seconds < 60:
        return f"{seconds}秒"

    elif seconds < 3600:
        minutes = math.floor(seconds / 60)
        remaining_seconds = seconds % 60
        return f"{minutes}分{remaining_seconds}秒"

    else:
        hours = math.floor(seconds / 3600)
        minutes = math.floor((seconds % 3600) / 60)
        return f"{hours}時間{minutes}分{seconds % 60}秒"


def split_text_by_time_intervals(json_data, split_duration=60) -> dict[str, dict[str, str]]:
    """与えられた JSON データを指定した時間間隔でテキストを分割し、各チャンクの情報を返す

    Parameters:
        json_data (list): JSON データのリスト
        split_duration (float): 区切りの秒数

    Returns:
        dict: 各チャンクの情報を含む辞書。キーはチャンクの番号、値はチャンクの情報を含む辞書
    """

    split_texts = list()
    current_text = ""
    current_start = 0
    current_end = split_duration

    for entry in json_data:
        start_time = entry["start"]
        text = entry["text"]

        # テキストが現在の範囲内に収まっているかを確認
        if start_time >= current_start + split_duration:
            split_texts.append({"text": current_text.strip(), "start": current_start, "end": current_end})
            current_text = ""
            current_start += split_duration
            current_end += split_duration

        # テキストを追加
        current_text += text

    # 最後の部分を追加
    if current_text:
        split_texts.append({"text": current_text.strip(), "start": current_start, "end": current_end})

    chunk_dict = {i: chunk_info for i, chunk_info in enumerate(split_texts)}

    return chunk_dict


def generate_video_response(llm: ChatOpenAI, question: str, content: str) -> str:
    system_template = "あなたは、質問者からの質問を回答するAIです。"
    human_template = """
        以下のテキストを元に「{question}」についての質問に答えてください。

        {document}
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    prompt_message_list = chat_prompt.format_prompt(
        question=question,
        document=content).to_messages()
    response = llm(prompt_message_list)

    return response


def generate_video_time_response(llm: ChatOpenAI, keyword: str, chunk_dict: dict[str, dict]) -> str:
    system_template = "あなたは、質問者からの質問を回答するAIです。"
    human_template = """
    キーワード: {keyword}

    jsonデータ:
    --------------------
    {chunk_dict}
    --------------------

    上記のjsonデータの中から、キーワード「{keyword}」と最も関連性が高いtextのインデックスを答えてください。

    回答の形式は
    「{keyword}の説明は{index}番です。」
    としてください。
    もしも、{keyword}の説明がない場合は「{keyword}の説明は動画内にありません。」としてください。
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    prompt_message_list = chat_prompt.format_prompt(
        keyword=keyword,
        chunk_dict=chunk_dict,
        index="インデックス").to_messages()
    response = llm(prompt_message_list)

    return response


def main() -> None:
    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"),
                     model_name="gpt-3.5-turbo-16k",
                     temperature=0)
    set_up_page()
    session_state = init_session_state(st.session_state)

    url = st.text_input("Youtube URL: ", key="input")
    if url:
        with st.spinner("Fetching Content ..."):
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language=['ja']
            )
            document = loader.load()
            video_id = document[0].metadata["source"]
            content = document[0].page_content
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["ja"])
            chunk_dict = split_text_by_time_intervals(transcript)

    if user_input := st.chat_input("聞きたいことを入力してね！"):
        session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Chatbot is typing ..."):
            additional_kwargs = get_question(llm, user_input)
            if additional_kwargs:
                if additional_kwargs["function_call"]["name"] == "generate_video_response":
                    question = json.loads(additional_kwargs["function_call"]["arguments"]).get("question")
                    response = generate_video_response(llm, question, content)
                    session_state.messages.append(AIMessage(content=response.content))

                elif additional_kwargs["function_call"]["name"] == "generate_video_time_response":
                    keyword = json.loads(additional_kwargs["function_call"]["arguments"]).get("keyword")
                    response = generate_video_time_response(llm, keyword, chunk_dict)
                    additional_kwargs = get_index(llm, response.content)
                    if additional_kwargs:
                        # TODO: ここでfunction callingを使用するかが悩ましい(third_response内には数字が入っているけど取得していない時がある)
                        index = json.loads(additional_kwargs["function_call"]["arguments"]).get("index")
                        if index is not None:
                            start = convert_seconds(chunk_dict[int(index)]["start"])
                            end = convert_seconds(chunk_dict[int(index)]["end"])
                            answer = f"{keyword}の説明は動画の{start}から{end}です。"
                            session_state.messages.append(AIMessage(content=answer))
                        else:
                            answer = f"私が思う{keyword}の説明に最も関連性が高いtextのインデックスは{index}番です。"
                            session_state.messages.append(AIMessage(content=answer))
                    else:
                        session_state.messages.append(AIMessage(content=response.content))
            else:
                response = llm(session_state.messages)
                session_state.messages.append(AIMessage(content=response.content))

    # チャット履歴の表示
    messages = session_state.get('messages', [])
    display_chat_history(messages)


if __name__ == '__main__':
    main()
