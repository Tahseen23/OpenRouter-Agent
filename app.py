import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.chat_models import ChatOpenAI
import os
import json
from langchain_community.agent_toolkits import JsonToolkit
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.json.tool import JsonSpec


handleGreetings = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that classify if {input} weather it is greetings or not if it is a grettings give response in dictionary with two keys greet:True and message:reply for that greeting if it is not then greet:False,message:None ",
    ),
    ("human", "{input}"),
])


def loadLLM():
  llm=ChatOpenAI(
    model_name='deepseek/deepseek-r1:free',
    api_key=os.getenv('MODEL_API'),
    base_url='https://openrouter.ai/api/v1'
  )
  return llm

def jsonAgent(file,llm):
  data=json.loads(file)
  print(data)
  print(isinstance(data,list))
  if isinstance(data, list):
      data_dict = {item['id']: item for item in data}
  else:
      data_dict = data 

  json_spec = JsonSpec(dict_=data_dict, max_value_length=4000)
  json_toolkit = JsonToolkit(spec=json_spec)

  # Create the JSON agent
  json_agent_executor = create_json_agent(
      llm=llm,
      toolkit=json_toolkit,
      verbose=True,
      agent_executor_kwargs={"handle_parsing_errors": True},
  )
  return json_agent_executor

def csvAgent(file,llm):
  template = """
  You are working with a pandas dataframe in Python. The dataframe name is `df`.

  Use the tools below to answer the question posed to you:

  - **python_repl_ast**: A Python shell for executing commands.
    - Input should be a valid Python command.
    - Ensure output is **not abbreviated** before using it.

  ### **Response Format**
  1. **Question**: {input}
  2. **Thought**: Your reasoning about the question.
  3. **Action**: The tool to use (`python_repl_ast`).
  4. **Action Input**: The Python command to execute.
  5. **Observation**: The result.
  6. Repeat as needed.
  7. **Final Answer**: The correct answer.

  **Begin!**
  Question: {input}
  {agent_scratchpad}
  """

  custom_prompt = PromptTemplate(input_variables=["input", "agent_scratchpad"], template=template)
  agent = create_csv_agent(
    llm=llm,
    path=file,
    verbose=True,
    agent_type="zero-shot-react-description",
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
    prompt=custom_prompt  
  )
  
  return agent

def responseAgentOne(agent,query):
  return agent.invoke({"input": "You have to conduct research and gather relevant information based on " +query+ " and the data provided"})

def responseAgentSecond(agent,query):
  return agent.invoke({"input": "You have to summarizes the  " +query+ " into a concise and conversational response for the user"})




st.title('Chat CSV/JSON')
uploadFile=st.file_uploader('Choose a file',type=['csv','json'])
if uploadFile :
  llm=loadLLM()
  fileType=os.path.splitext(uploadFile.name)[1]
  if fileType=='.json':
    file=uploadFile.getvalue().decode('utf-8').strip()
    if not file:
      st.error("Error: The uploaded JSON file is empty!")
    else:
      agent=jsonAgent(file,llm)
  elif fileType=='.csv':
    agent=csvAgent(uploadFile,llm)
  else:
    st.write('Please upload csv or json file')

if uploadFile:
  if 'messages' not in st.session_state:
    st.session_state.messages=[]
  
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
  
  if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    chain = handleGreetings | llm
    ai_msg = chain.invoke({'input':'{prompt}'})
    print(ai_msg)
    json_content = ai_msg.content.strip("```json").strip("```").strip()
    resp = json.loads(json_content)
    if not resp['greet']:
      responseOne = responseAgentSecond(agent,prompt)
      print(responseOne)
      if responseOne:
        responseSec=responseAgentSecond(agent,responseOne['output'])
      with st.chat_message("assistant"):
          st.markdown(responseOne['output'])
          st.markdown(responseSec['output'])
      st.session_state.messages.append({"role": "assistant", "content": [responseOne['output'],responseSec['output']]})
    else:
      with st.chat_message("assistant"):
          st.markdown(resp['message'])
      st.session_state.messages.append({"role": "assistant", "content": [resp['message']]})

  
