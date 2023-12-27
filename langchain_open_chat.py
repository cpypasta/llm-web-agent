import streamlit as st, os, requests, openai
from enum import Enum
from openrouter import OpenRouter
from typing import Any
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.lib.mutable_status_container import StatusContainer
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import Tool, AgentExecutor, load_tools, initialize_agent, AgentType, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.llms.ollama import Ollama
from langchain_core.language_models.llms import LLM
from langchain.chains import LLMChain

load_dotenv()

class ToolHandler(BaseCallbackHandler):
  status: StatusContainer
  tool_input: str
  
  def __init__(self, container: DeltaGenerator, logger: list[str]):
    self.container = container
    self.logger = logger
  def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs) -> None:
    with self.container:
      self.tool_name = serialized["name"]      
      self.status = st.status(f"{self.tool_name}...", expanded=False)
      self.tool_input = input_str
      with self.status:
        st.code(f"{input_str}")
  def on_tool_end(self, output: str, **kwargs: Any) -> Any:
    self.logger.extend([f"{self.tool_name} complete", self.tool_input])
    self.status.update(label=f"{self.tool_name} complete", state="complete")

class StreamHandler(BaseCallbackHandler):
  def __init__(self, container: DeltaGenerator, initial_text: str = ""):
    self.container = container
    self.text=initial_text
  def on_llm_new_token(self, token: str, **kwargs) -> None:
    processing_message = "PROCESSING" in token
    if not processing_message:
      self.text += token
      self.container.markdown(self.text)

class LLMType(Enum):
  OPENROUTER = "OpenRouter"
  OLLAMA = "Ollama"

def create_llm(provider: str, model: str, stream_handler: StreamHandler) -> LLM:
  if provider == LLMType.OLLAMA.value:
    client = Ollama(
      model=model,
      callbacks=[stream_handler]
    )  
  elif provider == LLMType.OPENROUTER.value:
    client = ChatOpenAI(
      model=model, 
      streaming=True, 
      callbacks=[stream_handler],
      api_key=os.getenv("OPENROUTER_API_KEY"),
      base_url="https://openrouter.ai/api/v1",
      default_headers={
        "HTTP-Referer": "https://streamlit.io/",
        "X-Title": "Streamlit OpenSource Chat"
      }
    )    
  return client

def create_agent(stream_handler: StreamHandler, tool_handler: ToolHandler) -> AgentExecutor:
  # client = ChatOpenAI(
  #   model="cognitivecomputations/dolphin-mixtral-8x7b", 
  #   streaming=False, 
  #   # callbacks=[stream_handler],
  #   api_key=os.getenv("OPENROUTER_API_KEY"),
  #   base_url="https://openrouter.ai/api/v1"
  # )
  client = Ollama(
    model="mistral",
    streaming=False,
    # callbacks=[stream_handler]
  )
  # tools = load_tools(["llm-math"], client, callbacks=[tool_handler])      
  tools = load_tools(["llm-math"], client)      
  agent = initialize_agent(tools, client, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
  agent.handle_parsing_errors = True
  return agent 

def render_chat_history() -> str:
  chat_history = ""
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])
      chat_history += f"{message['role']}: {message['content']}\n"
  return chat_history

@st.cache_data
def get_available_models(llm_provider: str) -> list[str]:
  if llm_provider == LLMType.OLLAMA.value:
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
      models = [m["name"] for m in response.json()["models"]]
    else: 
      models = []
  elif llm_provider == LLMType.OPENROUTER.value:
    response = requests.get("https://openrouter.ai/api/v1/models")
    if response.status_code == 200:
      models = [m["id"] for m in response.json()["data"]]
    else:
      models = []
  else:
    models = []  
  return models

if __name__ == "__main__":
  openai_key = os.getenv("OPENAI_API_KEY")
  
  st.title("OpenSource Chat")
  
  with st.sidebar:
    llm_provider = st.selectbox(
      "LLM Provider", 
      [LLMType.OLLAMA.value, LLMType.OPENROUTER.value], 
      index=0
    )
    models = get_available_models(llm_provider)
    llm_model = st.selectbox("LLM Model", models, index=0)
  
  is_new_conversation = "messages" not in st.session_state
  if is_new_conversation:
    st.session_state["messages"] = []
  else:
    chat_history = render_chat_history()
  
  if prompt := st.chat_input("What would you like to talk about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)
    with st.chat_message("assistant"):
      message_placeholder = st.empty()
      stream_handler = StreamHandler(message_placeholder)
      client = create_llm(llm_provider, llm_model, stream_handler)
      prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant."),
        ("user", """
        {chat_history}     
        user: {human_input}
        assistant:""")
      ])      
      chain = LLMChain(llm=client, prompt=prompt_template)  
      try:    
        full_response = chain.run(human_input=prompt, chat_history=chat_history)
      except Exception as e:
        print(e)
        full_response = stream_handler.text
      message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({
      "role": "assistant", 
      "content": full_response
    })