import streamlit as st, os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if __name__ == "__main__":
  st.title("🤖 OpenAI Chat")
  
  with st.sidebar:
    openai_key = st.text_input(
      "OpenAI API Key", 
      os.getenv("OPENAI_API_KEY"), 
      type="password",
      help="Will use `gpt-3.5-turbo`")
    if openai_key:
      os.environ["OPENAI_API_KEY"] = openai_key
      
  client = OpenAI()
  
  if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
    
  if "messages" not in st.session_state:
    st.session_state["messages"] = []

  # chat history
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])
  
  if openai_key:  
    if prompt := st.chat_input("What is up?"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
        st.markdown(prompt)
      with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
          model=st.session_state.openai_model,
          messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
          ],
          stream=True
        ):
          full_response += (response.choices[0].delta.content or "")
          message_placeholder.markdown(full_response + " ")
        message_placeholder.markdown(full_response)
      
      st.session_state.messages.append({"role": "assistant", "content": full_response})
  else:
    st.warning("Please enter an OpenAI API Key.")