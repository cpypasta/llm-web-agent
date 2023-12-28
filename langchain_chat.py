import streamlit as st, os
from streamlit.delta_generator import DeltaGenerator
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

class StreamHandler(BaseCallbackHandler):
  def __init__(self, container: DeltaGenerator, initial_text: str = ""):
    self.container = container
    self.text=initial_text
  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

if __name__ == "__main__":
  st.title("🦜🔗 LangChain Chat")
    
  with st.sidebar:
    openai_key = st.text_input(
      "OpenAI API Key", 
      os.getenv("OPENAI_API_KEY"), 
      type="password",
      help="Will use `gpt-3.5-turbo`")
    if openai_key:
      os.environ["OPENAI_API_KEY"] = openai_key    
    
  if "messages" not in st.session_state:
    st.session_state["messages"] = []

  # chat history
  chat_history = ""
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])
      chat_history += f"{message['role']}: {message['content']}\n"
  
  if openai_key:
    if prompt := st.chat_input("What is up?"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
        st.markdown(prompt)
      with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)
        client = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, callbacks=[stream_handler])
        prompt_template = ChatPromptTemplate.from_messages([
          ("system", "You are an AI assistant."),
          ("user", """
          {chat_history}     
          user: {human_input}
          assistant:""")
        ])      
        chain = LLMChain(llm=client, prompt=prompt_template)      
        full_response = chain.run(human_input=prompt, chat_history=chat_history)
        message_placeholder.markdown(full_response)
      
      st.session_state.messages.append({"role": "assistant", "content": full_response})
  else:
    st.warning("Please enter an OpenAI API Key.")      