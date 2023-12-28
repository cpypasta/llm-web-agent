import streamlit as st, os
from typing import Any
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.lib.mutable_status_container import StatusContainer
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import Tool, AgentExecutor, load_tools
from langchain_experimental.tools import PythonREPLTool
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

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
    self.text += token
    self.container.markdown(self.text)

if __name__ == "__main__":
  st.title("🤖 OpenAI Agent Chat")
  
  with st.sidebar:
    st.markdown("""
      This is a Chatbot that uses a LangChain Agent and the OpenAI API. The agent has access to the following tools:
      1. `terminal`: the terminal
      2. `ddg-search`: DuckDuckGo Search
      3. `python`: a Python REPL
      4. `llm-math`: adds math capabilities
    """)
    openai_key_input = st.text_input(
      "OpenAI API Key", 
      os.getenv("OPENAI_API_KEY"), 
      type="password", 
      help="Will use `gpt-3.5-turbo`"
    )
  
  if openai_key_input:
    if "messages" not in st.session_state:
      st.session_state["messages"] = []

    # chat history
    chat_history = ""
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        if "tool" in message and len(message["tool"]) > 0:
          with st.status(message["tool"][0], state="complete", expanded=False):
            st.code(message["tool"][1])
        st.markdown(message["content"])
        chat_history += f"{message['role']}: {message['content']}\n"    
    
    if prompt := st.chat_input("what tools do you have access to?"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
        st.markdown(prompt)
      with st.chat_message("assistant"):
        tool_placeholder = st.empty()
        tool_input = []
        tool_handler = ToolHandler(tool_placeholder, tool_input)
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)
        client = ChatOpenAI(
          model="gpt-3.5-turbo", 
          streaming=True, 
          callbacks=[stream_handler],
          api_key=openai_key_input
        )
        tools = load_tools(["terminal", "ddg-search", "llm-math"], client, callbacks=[tool_handler])
        tools.append(PythonREPLTool(name="python", callbacks=[tool_handler]))
        openai_tools = [format_tool_to_openai_function(t) for t in tools]
        client_with_tools = client.bind(functions=openai_tools)
        prompt_template = ChatPromptTemplate.from_messages([
          ("system", "You are an AI assistant."),
          ("user", """
          {chat_history}     
          user: {human_input}
          assistant:"""),
          MessagesPlaceholder(variable_name="agent_scratchpad")
        ])      
        chain = (
          {
            "human_input": lambda x: x["human_input"],
            "chat_history": lambda x: x["chat_history"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"])
          }
          | prompt_template | client_with_tools | OpenAIFunctionsAgentOutputParser()
        )   
        agent = AgentExecutor(agent=chain, tools=tools, verbose=True, handle_parsing_errors=True)
        full_response = agent.invoke({"human_input": prompt, "chat_history": chat_history})
        full_response = full_response["output"]
        message_placeholder.markdown(full_response)
      
      st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response, 
        "tool": tool_input
      })
  else:
    st.warning("Please enter an OpenAI API Key.")