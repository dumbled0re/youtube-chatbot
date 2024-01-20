import os

import streamlit as st
from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"),
                     temperature=0)

    # ページ上部の部分
    st.set_page_config(
        page_title="Youtube chatbot",
        page_icon="🤗"
    )
    st.header("Youtube chatbot 🤗")

    # チャット履歴の初期化
    # つまりここは何でもいいのか(最初しか実行されないぽいな)
    # TODO: st.session_stateが空という条件のほうがいいな
    if "messages" not in st.session_state:
        print(f" ###### messages #######: {st.session_state}")
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # ユーザーの入力を監視
    # これだけでchatUIになる
    if user_input := st.chat_input("テキスト入力"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("生成中..."):
            response = llm(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

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
        # else:  # isinstance(message, SystemMessage):
        #     st.write(f"System message: {message.content}")


if __name__ == '__main__':
    main()
