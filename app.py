import streamlit as st
import numpy as np
from functions import handle_query, generate_answer, generate_cover_letter, certif_recommendation, get_model # Import de la fonction
from langchain_groq import ChatGroq


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"





# db =load_chroma_db()

st.set_page_config(page_title="InfoBot UCBL", page_icon=":robot_face")
st.title('🎓 InfoBot UCBL')
st.write('Votre compagnon intelligent pour tout savoir sur le Département Informatique de l’Université Claude Bernard Lyon 1 !')


# prompt = st.chat_input("Say something")
# if prompt:
#     st.write(f"User has sent the following prompt: {prompt}")


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
    llm = get_model()
    tools = [generate_answer, generate_cover_letter, certif_recommendation] #
    response = handle_query(query, llm, tools)
    print("This is the response : " , response)
    
    # Display assistant response in chat message container 
    with st.chat_message("assistant"):
        st.markdown(response)


    # Add assistant response to chat history 
    st.session_state.messages.append({"role" : "assistant", "content": response})

    