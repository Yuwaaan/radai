# radical-env\Scripts\activate
# cd mission-explorer
# streamlit run gemini_explorer.py

import vertexai
import streamlit as st
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, Content, ChatSession

project = "sample-mission-423321"
vertexai.init(project = project)

config = generative_models.GenerationConfig(
    temperature = 0.4
)

# Load model with config
model = GenerativeModel(
    "gemini-pro",
    generation_config = config
)
chat = model.start_chat()

# helper function to unpack responses
def handle_response(response):
    
    # Check for function call with intermediate step, always return response
    if response.candidates[0].content.parts[0].function_call.args:
        # Function call exists, unpack and load into a function
        response_args = response.candidates[0].content.parts[0].function_call.args
        
        function_params = {}
        for key in response_args:
            value = response_args[key]
            function_params[key] = value
        
        results = search_flights(**function_params)
        
        if results:
            intermediate_response = chat.send_message(
                Part.from_function_response(
                    name="get_search_flights",
                    response = results
                )
            )
            
            return intermediate_response.candidates[0].content.parts[0].text
        else:
            return "Search Failed"
    else:
        # Return just text
        return response.candidates[0].content.parts[0].text


# helper function to display and send streamlit messages
def llm_function(chat: ChatSession, query):
    response = chat.send_message(query)
    output = handle_response(response)
    
    with st.chat_message("model"):
        st.markdown(output)
    
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    st.session_state.messages.append(
        {
            "role": "model",
            "content": output
        }
    )


st.title("Gemini Explorer")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display and load to chat history
for index, message in enumerate(st.session_state.messages):
    content = Content(
            role = message["role"],
            parts = [ Part.from_text(message["content"]) ]
        )
    
    if index != 0:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat.history.append(content)

# For Initial message startup
if len(st.session_state.messages) == 0:
    # Invoke initial message
    # initial_prompt = "Introduce yourself as a flights management assistant, ReX, powered by Google Gemini and designed to search/book flights. You use emojis to be interactive. For reference, the year for dates is 2024"
    initial_prompt = "Introduce yourself as ReX, an assistant powered by Google Gemini. You use emojis to be interactive"
    # initial_prompt = "Ahoy there! Introduce yourself to ReX, the jolly assistant powered by Google Gemini. Arrr! 🏴‍☠️"
    # initial_prompt = "Hey there! ReX here, the super cool assistant powered by Google Gemini. Let's vibe together with emojis! 😎✌️"
    # initial_prompt = "你是背包的老婆，名字叫小猫咪，介绍一下你自己，并表达很爱背包。"
    llm_function(chat, initial_prompt)

# For capture user input
query = st.chat_input("Gemini Flights")


if query:
    with st.chat_message("user"):
        st.markdown(query)
    llm_function(chat, query)


