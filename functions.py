import os
import re
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from templates import get_qa_template, get_cv_template, get_recommendation_template
from langchain.agents import tool
from typing import List
from langchain_core.messages import HumanMessage, ToolMessage
from typing import List
from langchain_groq import ChatGroq



# Vérifier si USER_AGENT existe
user_agent = os.environ.get("USER_AGENT")

if user_agent:
    print(f"USER_AGENT existe déjà : {user_agent}")
else:
    # Définir USER_AGENT s'il n'existe pas
    os.environ["USER_AGENT"] = "MyApp"
    print("USER_AGENT n'existait pas. Il a été créé avec la valeur : MyApp")


def get_model():
  GROQ_API_KEY="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"

  os.environ["GROQ_API_KEY"] = GROQ_API_KEY
  llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
  return llm


# Charger la base Chroma existante
def load_chroma_db():

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    CHROMA_DB_PATH = "/content/drive/MyDrive/LLM/CC2-LLM/chroma_combined_multi"
    # Vérifiez si le dossier Chroma existe
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Le dossier '{CHROMA_DB_PATH}' n'existe pas.")
        return None
    else:
        print(f"Le dossier '{CHROMA_DB_PATH}' a été trouvé.")

    # Charger la base de données Chroma
    db_all = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

    # Vérifiez si la base contient des documents
    if db_all._collection.count() == 0:
        print("Le dossier Chroma est vide, aucun document trouvé.")
    else:
        print(f"Le dossier Chroma contient {db_all._collection.count()} documents.")
    
    return db_all

def load_db_certif():

  embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  CHROMA_certif_PATH = "/content/drive/MyDrive/LLM/CC2-LLM/chroma_certif"
  # Vérifiez si le dossier Chroma existe
  if not os.path.exists(CHROMA_certif_PATH):
      print(f"Le dossier '{CHROMA_certif_PATH}' n'existe pas.")
      return None
  else:
      print(f"Le dossier '{CHROMA_certif_PATH}' a été trouvé.")

  # Charger la base de données Chroma
  db_certif = Chroma(persist_directory=CHROMA_certif_PATH, embedding_function=embedding_function)

  # Vérifiez si la base contient des documents
  if db_certif._collection.count() == 0:
      print("Le dossier Chroma est vide, aucun document trouvé.")
  else:
      print(f"Le dossier Chroma contient {db_certif._collection.count()} documents.")
  
  return db_certif


def get_ue_context(user_question: str):

  # Étendre la question utilisateur avec des sous-questions
  question_etendue = f"""
  {user_question}.
  Quelles sont les Unités d'Enseignement (UE) de cette formation ?
  Quelles sont les blocs de cette formation ?
  """
  db = load_chroma_db()
  retriever = db.as_retriever()
  relevent_docs = retriever.invoke(question_etendue)
  contexte_formation = " ".join(doc.page_content for doc in relevent_docs)

  return contexte_formation

def get_certif_context(user_question: str, contexte_formation: str):
  # contexte_formation = get_ue_context(user_question)

  question_certificats = f"""
  En vous basant sur les Unités d'Enseignement (UE) suivantes :
  {contexte_formation}

  Recherchez les certificats qui couvrent les compétences enseignées dans ces UEs et expliquez pourquoi chaque certificat est pertinent.
  """
  certificats_db = load_db_certif()
  print("---------------------------------------------------------certificats_db", certificats_db._collection.count() )
  certificats_retriever = certificats_db.as_retriever()
  certificats_recommandes = certificats_retriever.invoke(question_certificats)

  # certificats_recommandes = certificats_retriever.invoke(contexte_formation)
  if not certificats_recommandes:
        return "Je n'ai trouvé aucun certificat pertinent pour cette formation."

  # Joindre les textes des certificats recommandés en une seule chaîne de caractères
  contexte_certificats = " ".join(doc.page_content for doc in certificats_recommandes)

  return contexte_certificats


#############Tools#################################
@tool
def generate_answer(question: str) -> str:

  """Utilisez ce tool pour générer une réponse aux questions générales liées à l'université, aux formations, aux laboratoires ou aux contacts. Appelez ce tool également en cas de doute sur le choix du tool à utiliser pour répondre à la requête.

    Args:
        question: The user's question to be answered.
    """
  template = get_qa_template()

  # Charger la base de données Chroma
  db = load_chroma_db()
  retriever = db.as_retriever()
  relevent_docs = retriever.invoke(question)
  print("################## QnA Relevent Docs##################")
  print(relevent_docs)
  print("################################################")

  prompt = ChatPromptTemplate.from_template(template)
  prompt_with_context = prompt.format(context=relevent_docs, question=question)
  # print(prompt_with_context)
  llm = get_model()
  response = llm.invoke(prompt_with_context)
  
  return response

@tool
def generate_cover_letter(user_question: str):
  """Generate the cover letter when the user ask about a cover letter.
  
    Args:
        question: The user's question to be answered.
    """

  template = get_cv_template()

  db = load_chroma_db()
  retriever = db.as_retriever()

  user_context = retriever.invoke(user_question)

  prompt_prog = f"""
  En te basant sur le contexte, quelle est la formation mentionnée dans cette question : {user_question}.
  Formule la réponse sous la forme de "Le nom complet de la formation est : <nom de la formation>".

  Contexte : {user_context}
  """
  llm = get_model()

  program = llm.invoke(prompt_prog)

  question_resp = "Qui est le responsable de cette formation ?"

  question_link = "Quel est le lien vers le site web de cette formation ?"

  question_scol = "Quelle est l'adresse physique du département de cette formation ? Quelle est l'adresse mail de la scolarité ?"

  relevant_docs = retriever.invoke(user_question + program.content + prompt_prog + question_resp + question_link + question_scol)

  prompt = ChatPromptTemplate.from_template(template)
  prompt_with_context = prompt.format(context=relevant_docs, question=user_question)
  response = llm.invoke(prompt_with_context)

  return response

@tool 
def certif_recommendation(user_question):
  """Useful when the user asks for certificat recommendations.

    Args:
        question: The user's question to be answered.
        llm : The model to use 
    """
  template = get_recommendation_template()

  context_ue = get_ue_context(user_question)
  print("-----------------------------------------------------------")
  print(context_ue)

  context_certif = get_certif_context(user_question, context_ue)
  print("-----------------------------------------------------------")
  print(context_certif)

  prompt = ChatPromptTemplate.from_template(template)
  prompt_with_context = prompt.format(context_formation=context_ue, context_certificats= context_certif, question=user_question)

  llm = get_model()
  response = llm.invoke(prompt_with_context)

  return response



def handle_query(query  : str, llm, tools: List):
  """ 
  Handle a user query by invoking tools and the language model.

  Args: 
      query (str): The user's query to process.
      llm: The large language model instance.
      tools (list): List of available tools for the language model to use.

  Returns:
      str: The final response from the language model.
  """

  # Bind tools to the LLM
  llm_with_tools = llm.bind_tools(tools)

  # Initialize conversation messages
  messages = [HumanMessage(query)]
  
  # Invoke the LLM with the conversation messages and capture the intial response (Tool to use)
  ai_msg = llm_with_tools.invoke(messages)
  # print("**********************This is ai msg", ai_msg)


  # Append the AI's initial response to the human message
  messages.append(ai_msg)
  # print("**********************messages", messages)

  # Process tool
  for tool_call in ai_msg.tool_calls:
    # print("***************************** tool call", tool_call )
    # print("***************************** tool_call[name]", tool_call["name"] )

    # Match tool name to its corresponding function
    selected_tool = {
        "generate_answer": generate_answer,
        "generate_cover_letter": generate_cover_letter,
        "certif_recommendation": certif_recommendation

    }.get(tool_call["name"].lower())
    
    # print("***************************** selected_tool", selected_tool )

    # If the tool exists, invoke it with the provided arguments
    tool_output = selected_tool.invoke(tool_call["args"])

    # print("***************************** tool_output", tool_output )
    # print("**********************************************")

    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    print("***************************** messages", messages )
    print("**********************************************")

   # Get the final response after processing tool outputs
    final_response = tool_output.content
    print("This issssssss the final response!!", final_response)
    return final_response
    
