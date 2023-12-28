import streamlit as st, os, requests, together
from enum import Enum
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
  LMSTUDIO = "LM Studio"
  TOGETHER = "together.ai"

def create_llm(provider: str, model: str, temp: int, max_tokens: int, stream_handler: StreamHandler) -> LLM:
  if provider == LLMType.OLLAMA.value:
    client = Ollama(
      model=model,
      temperature=temp,
      callbacks=[stream_handler]
    )  
  else:
    if provider == LLMType.OPENROUTER.value:
      api_key=os.getenv("OPENROUTER_API_KEY")
      base_url="https://openrouter.ai/api/v1"
    elif provider == LLMType.LMSTUDIO.value:
      api_key="key"
      base_url="http://127.0.0.1:1234/v1"        
    elif provider == LLMType.TOGETHER.value:
      api_key=os.getenv("TOGETHER_API_KEY")
      base_url="https://api.together.xyz/v1"
    client = ChatOpenAI(
      model=model, 
      streaming=True, 
      callbacks=[stream_handler],
      temperature=temp,
      max_tokens=max_tokens,
      api_key=api_key,
      base_url=base_url
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
def get_available_models(llm_provider: str, openrouter_key: str, together_key: str) -> list[str]:
  if llm_provider == LLMType.OLLAMA.value:
    response = requests.get("http://127.0.0.1:11434/api/tags")
    if response.status_code == 200:
      models = [m["name"] for m in response.json()["models"]]
    else: 
      models = []
  elif llm_provider == LLMType.OPENROUTER.value and openrouter_key:
    response = requests.get("https://openrouter.ai/api/v1/models")
    if response.status_code == 200:
      models = [m["id"] for m in response.json()["data"]]
    else:
      models = []
  elif llm_provider == LLMType.LMSTUDIO.value:
    response = requests.get("http://127.0.0.1:1234/v1/models")
    if response.status_code == 200:
      models = [m["id"] for m in response.json()["data"]]
    else:
      models = []
  elif llm_provider == LLMType.TOGETHER.value and together_key:
    models = [m["name"] for m in together.Models.list()]
  else:
    models = []  
  return models

def ollama_is_available() -> bool:
  try:
    response = requests.get("http://127.0.0.1:11434/api/tags")
    return response.status_code == 200
  except:
    return False

def lm_studio_is_available() -> bool:
  try:
    response = requests.get("http://127.0.0.1:1234/v1/models")
    return response.status_code == 200
  except:
    return False

@st.cache_data
def get_available_providers() -> list[str]:
  providers = []
  if ollama_is_available():
    providers.append(LLMType.OLLAMA.value)
  if lm_studio_is_available():
    providers.append(LLMType.LMSTUDIO.value)
  providers.extend([LLMType.OPENROUTER.value, LLMType.TOGETHER.value])
  return providers

if __name__ == "__main__":
  openai_key = os.getenv("OPENAI_API_KEY")
  
  st.title("🤖 OpenSource Chat")
  
  with st.expander("About App", expanded=False):
    st.markdown("""This app is a playground for chatting with various OpenSource large language models. In order to use local providers, you should run this Streamlit app locally.""")
    st.markdown("The following hosting providers are supported:")
    st.dataframe([
      {"Provider": "OpenRouter", "type": "online", "port": "None"},
      {"Provider": "Together.ai", "type": "online", "port": "None"},
      {"Provider": "Ollama", "type": "local", "port": "11434"},
      {"Provider": "LM Studio", "type": "local", "port": "1234"},
    ], hide_index=True)
    st.warning("Not all models are chat models. Moreover, every model is different, so expect different results.")
  
  with st.sidebar:
    llm_provider = st.selectbox(
      "LLM Provider", 
      get_available_providers(), 
      index=0
    )
    if llm_provider == LLMType.OPENROUTER.value:
      openrouter_key = st.text_input("OpenRouter API Key", os.getenv("OPENROUTER_API_KEY"), type="password")
      if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
    elif llm_provider == LLMType.TOGETHER.value:
      together_key = st.text_input("Together.ai API Key", os.getenv("TOGETHER_API_KEY"), type="password")
      if together_key:
        os.environ["TOGETHER_API_KEY"] = together_key
        together.api_key = together_key
      
    models = get_available_models(llm_provider, openrouter_key, together_key)
    llm_model = st.selectbox("LLM Model", models, index=0)
    with st.expander("Options"):
      system_prompt = st.text_area("System Prompt", "You are an AI assistant.")
      llm_temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
      llm_output_tokens = st.slider("Max Output Tokens", 1024, 4096, 1024, 512, help="Does not work for Ollama.")
  
  is_new_conversation = "messages" not in st.session_state
  if is_new_conversation:
    st.session_state["messages"] = []
  else:
    chat_history = render_chat_history()
  
  if len(models) > 0:
    if prompt := st.chat_input("What would you like to talk about?"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
        st.markdown(prompt)
      with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)
        client = create_llm(llm_provider, llm_model, llm_temp, llm_output_tokens, stream_handler)
        prompt_template = ChatPromptTemplate.from_messages([
          ("system", system_prompt),
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
          if stream_handler.text:
            full_response = stream_handler.text
          else:
            full_response = f"ERROR:{e}"
        message_placeholder.markdown(full_response)
      
      if not full_response.startswith("ERROR:"):
        st.session_state.messages.append({
          "role": "assistant", 
          "content": full_response
        })
  else:
    st.warning("Please provide the necessary API key.")