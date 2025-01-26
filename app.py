import streamlit as st
from functions import handle_query, generate_answer, generate_cover_letter, certif_recommendation, get_model # Import de la fonction

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"





st.set_page_config(page_title="InfoBot UCBL", page_icon=":robot_face")
st.title('🎓 InfoBot UCBL')
st.write('Votre compagnon intelligent pour tout savoir sur le Département Informatique de l’Université Claude Bernard Lyon 1 !')



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Posez votre question ici...")
# React to user input
if query :
    # Display user message in chat message contaier
    with st.chat_message("user"):
        st.markdown(query)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get the LLM to generate responses
    llm = get_model()
    
    # Define the list of tools available for handling different types of queries
    tools = [generate_answer, generate_cover_letter, certif_recommendation] 
    
    # Call a function to handle the query, which uses the LLM and tools to generate a response
    response = handle_query(query, llm, tools)
    
    # Display assistant response in chat message container 
    with st.chat_message("assistant"):
        st.markdown(response)


    # Add assistant response to chat history 
    st.session_state.messages.append({"role" : "assistant", "content": response})

    