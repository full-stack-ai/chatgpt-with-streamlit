import streamlit as st
import os
from chatgpt_with_streamlit.app.streamlit_utils import stdout_streaming, load_chat_history, save_chat_history, remove_file_extension
from chatgpt_with_streamlit.GetLanguageModel.call_llm import get_chat_llm_chain

# Streamlit application
def run():
    """Streamlit application run function"""

    # messages init
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar components
    ## Handle message history on the sidebar
    message_history_files = [file for file in os.listdir() if file.endswith("_message.json")] # assume that stored messages end with _message.json

    local_llm_filename_list = os.listdir("./models")
    local_llms = [file for file in local_llm_filename_list if file.endswith("gguf")]
    list_of_llms = local_llms + ["openai"]
    sidebar = st.sidebar
    with sidebar:
        # Selectbox for LLM
        llm_option = st.selectbox("Select an LLM Option:", list_of_llms)
        # Select LLM option [Openai, local LLMs]
        save_chat_btn = st.button("Save Chat", use_container_width=True)
        st.markdown("""___""")
        for file in message_history_files:
            file_topic = remove_file_extension(file)
            if st.button(file_topic, use_container_width=True):
                pass
                st.session_state.messages = load_chat_history(file)
        
            
    # Call the save chat history function
    if save_chat_btn:
        if len(st.session_state.messages) != 0:
            save_chat_history(st.session_state.messages)
            pass
        st.session_state.messages = []


    # Image icon similar to openAI
    img_path = "./static/logo.jpg"
    cols = st.columns(10)
    with cols[4]:
        st.image(img_path, width=100)
    

    st.markdown("<h1 style='text-align: center;'>How can I help you today?</h1>",
                unsafe_allow_html=True)
    
    # Populate messages in the chat message component everytime the streamlit is run
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("Message ChatGPT..."):
        # User message handling
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # AI message  handling
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                output = st.empty()
                # Function to pass stdout to the streamlit IO
                # LLMs may stream on the stdout
                # Call streamlit utils function
                with stdout_streaming(output.info):
                    # Call LLM chain and invoke the LLM model to generate response
                    chain = get_chat_llm_chain(llm_option=llm_option)
                    response = chain.invoke(user_input) # Right now only the last message is passed

                    st.session_state.messages.append(
                        {"role": "ai", "content": response["text"]}
                    )
    return


if __name__ == "__main__":
    run()