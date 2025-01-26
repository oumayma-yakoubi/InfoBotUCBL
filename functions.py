import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from templates import get_qa_template, get_cv_template, get_recommendation_template
from langchain.agents import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from typing import List



# Check if USER_AGENT exists in environment variables
user_agent = os.environ.get("USER_AGENT")

if user_agent:
    print(f"USER_AGENT existe déjà : {user_agent}")
else:
    # If USER_AGENT doesn't exist, set it with a default value
    os.environ["USER_AGENT"] = "MyApp"
    print("USER_AGENT n'existait pas. Il a été créé avec la valeur : MyApp")


def get_model():
  GROQ_API_KEY=""

  os.environ["GROQ_API_KEY"] = GROQ_API_KEY
  llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
  return llm


# Load the existing Chroma database
def load_chroma_db():

    # Define the embedding function 
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    CHROMA_DB_PATH = "chroma_combined_multi"
    # Check if the Chroma directory exists
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Le dossier '{CHROMA_DB_PATH}' n'existe pas.")
        return None
    else:
        print(f"Le dossier '{CHROMA_DB_PATH}' a été trouvé.")

    # Load the Chroma database using the given path and embedding function
    db_all = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

    # Check if the database contains any documents
    if db_all._collection.count() == 0:
        print("Le dossier Chroma est vide, aucun document trouvé.")
    else:
        print(f"Le dossier Chroma contient {db_all._collection.count()} documents.")
    
    # Return the loaded Chroma database
    return db_all

def load_db_certif():

  embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  CHROMA_certif_PATH = "chroma_certif"

  if not os.path.exists(CHROMA_certif_PATH):
      print(f"Le dossier '{CHROMA_certif_PATH}' n'existe pas.")
      return None
  else:
      print(f"Le dossier '{CHROMA_certif_PATH}' a été trouvé.")

  db_certif = Chroma(persist_directory=CHROMA_certif_PATH, embedding_function=embedding_function)

  if db_certif._collection.count() == 0:
      print("Le dossier Chroma est vide, aucun document trouvé.")
  else:
      print(f"Le dossier Chroma contient {db_certif._collection.count()} documents.")
  
  return db_certif


def get_ue_context(user_question: str):

  # Extend the user question with additional sub-questions to get relevant context
  question_etendue = f"""
  {user_question}.
  Quelles sont les Unités d'Enseignement (UE) de cette formation ?
  Quelles sont les blocs de cette formation ?
  """
  db = load_chroma_db()
  retriever = db.as_retriever()
  
  # Invoke the retriever with the extended question to fetch relevant documents
  relevent_docs = retriever.invoke(question_etendue)
  contexte_formation = " ".join(doc.page_content for doc in relevent_docs)
  
  # Return the aggregated context (formation details) found in the relevant documents
  return contexte_formation

def get_certif_context(contexte_formation: str):

  question_certificats = f"""
  En vous basant sur les Unités d'Enseignement (UE) suivantes :
  {contexte_formation}

  Recherchez les certificats qui couvrent les compétences enseignées dans ces UEs et expliquez pourquoi chaque certificat est pertinent.
  """
  certificats_db = load_db_certif()
  certificats_retriever = certificats_db.as_retriever()
  certificats_recommandes = certificats_retriever.invoke(question_certificats)

  if not certificats_recommandes:
        return "Je n'ai trouvé aucun certificat pertinent pour cette formation."

  contexte_certificats = " ".join(doc.page_content for doc in certificats_recommandes)

  return contexte_certificats

####################################
################Tools###############
####################################
@tool
def generate_answer(question: str) -> str:

  """Utilisez ce tool pour générer une réponse aux questions générales liées à l'université, aux formations, aux laboratoires ou aux contacts. Appelez ce tool également en cas de doute sur le choix du tool à utiliser pour répondre à la requête.

    Args:
        question: The user's question to be answered.
    """
  # Get the QA template for generating the response
  template = get_qa_template()

  # Load the Chroma database for retrieving relevant documents
  db = load_chroma_db()
  retriever = db.as_retriever()
  
  # Invoke the retriever to search for relevant documents based on the user's question
  relevent_docs = retriever.invoke(question)

  prompt = ChatPromptTemplate.from_template(template)
  prompt_with_context = prompt.format(context=relevent_docs, question=question)

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

  context_certif = get_certif_context(user_question, context_ue)

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

  # Append the AI's initial response to the human message
  messages.append(ai_msg)

  # Process tool
  for tool_call in ai_msg.tool_calls:

    # Match tool name to its corresponding function
    selected_tool = {
        "generate_answer": generate_answer,
        "generate_cover_letter": generate_cover_letter,
        "certif_recommendation": certif_recommendation

    }.get(tool_call["name"].lower())
    
    # If the tool exists, invoke it with the provided arguments
    tool_output = selected_tool.invoke(tool_call["args"])

    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    
   # Get the final response after processing tool outputs
    final_response = tool_output.content
    return final_response
    
