import streamlit as st
import json
import random

def main():
    sidebar = st.sidebar
    with sidebar:
        reset_chat_btn = st.button("Reset Chat")
    
    if reset_chat_btn:
        if len(st.session_state.messages) != 0:
            rand_num = random.randint(0,1000000)
            with open(f"messages_{rand_num}.json") as file:
                json.dump(st.session_state.messages, file)
        st.session_state.messages = []



    st.title("ChatGPT")
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
            ai_message = f"Your number is"
            st.markdown(ai_message)
        st.session_state.messages.append(
            {"role": "ai", "content": ai_message}
        )



if __name__ == "__main__":
    main()
