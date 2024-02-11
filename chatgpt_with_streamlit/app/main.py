import streamlit as st
from chatgpt_with_streamlit.app.streamlit_utils import save_chat_history, st_stream_from_stdout

from chatgpt_with_streamlit.GetLanguageModel.call_llm import get_chain


def run():
    sidebar = st.sidebar
    with sidebar:
        reset_chat_btn = st.button("Reset Chat", use_container_width=True)

    if reset_chat_btn:
        if len(st.session_state.messages) != 0:
            save_chat_history(st.session_state.messages)
        st.session_state.messages = []

    # Your image file path
    image_path = './static/logo.jpg'
    cols = st.columns(10)
    with cols[4]:
        st.image(image_path, width=100)

    st.markdown("<h1 style='text-align: center;'>How can I help you today?</h1>",
                unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Message ChatGPT..."):
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                output = st.empty()
                with st_stream_from_stdout(output.info):
                    chain = get_chain()
                    response = chain.invoke(user_input)
                    st.session_state.messages.append(
                        {"role": "ai", "content": response['text']}
                    )


if __name__ == "__main__":
    run()
