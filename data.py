from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import RecursiveUrlLoader
from markdownify import markdownify
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup, SoupStrainer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import ChatGroq
import json
import re
import os

os.environ['USER_AGENT'] = 'Agent'

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\+", " ", soup.text).strip()

# This function loads data and returns all_docs , ue_docs, certif_docs
def load_documents():
  load_dep = RecursiveUrlLoader("https://fst-informatique.univ-lyon1.fr/departement", max_depth=2, extractor=bs4_extractor,)
  load_formation = RecursiveUrlLoader("https://fst-informatique.univ-lyon1.fr/formation", max_depth=2, extractor=bs4_extractor,)
  load_masters = RecursiveUrlLoader("http://master-info.univ-lyon1.fr/", max_depth=2, extractor=bs4_extractor,)
  load_licence = WebBaseLoader(web_path=("http://licence-info.univ-lyon1.fr/",))
  load_DISS = WebBaseLoader(web_path=("http://master-info.univ-lyon1.fr/DISS/",))
  load_MEEF = WebBaseLoader(web_path=("http://master-info.univ-lyon1.fr/MEEF-NSI/",))
  load_SRS = WebBaseLoader(web_path=("http://master-info.univ-lyon1.fr/SRS/",))
  load_MSCIENCES_HUMAINES = WebBaseLoader(web_path=("https://offre-de-formations.univ-lyon1.fr/mention-72/sciences-de-l-information-et-des-bibliotheques.html",),)
  load_dist= WebBaseLoader(web_path=(" http://dist.univ-lyon1.fr/",),) 
  load_Mtechnico_commerciale = WebBaseLoader(web_path=("https://offre-de-formations.univ-lyon1.fr/parcours-1259/m2-ingenierie-technico-commerciale.html",),)
  load_MBiof = WebBaseLoader(web_path=("https://www.bioinfo-lyon.fr/",))

  #Relations Entreprises
  load_REntreprise = RecursiveUrlLoader("https://fst-informatique.univ-lyon1.fr/relations-entreprises", max_depth=2, extractor=bs4_extractor,)

  # Laboratoires
  # --> Laboratoire https://fst-informatique.univ-lyon1.fr/laboratoires
  load_Laboratoires = WebBaseLoader(web_path=("https://fst-informatique.univ-lyon1.fr/laboratoires",))
  load_liris = WebBaseLoader(web_path=("https://www.univ-lyon1.fr/recherche/entites-de-recherche-et-plateformes-technologiques/laboratoire-dinformatique-en-images-et-systemes-dinformation-liris",))
  load_elico = WebBaseLoader(web_path=("https://www.univ-lyon1.fr/recherche/entites-de-recherche-et-plateformes-technologiques/equipe-de-recherche-de-lyon-en-sciences-de-linformation-et-de-la-communication-elico",))
  load_lip = WebBaseLoader(web_path=("https://www.univ-lyon1.fr/recherche/entites-de-recherche-et-plateformes-technologiques/laboratoire-de-linformatique-du-parallelisme-lip",))
  load_eric = WebBaseLoader(web_path=("https://www.univ-lyon1.fr/recherche/entites-de-recherche-et-plateformes-technologiques/entrepots-representation-et-ingenierie-des-connaissances-eric",))

  loader_all = [load_dep, load_formation, load_masters, load_licence, load_DISS, load_MEEF, load_SRS, load_MSCIENCES_HUMAINES, load_dist
  load_Mtechnico_commerciale, load_MBiof, load_REntreprise, load_Laboratoires, load_liris, load_elico, load_lip, load_eric]

  # Load all data
  all_docs = []
  for load in loader_all:
    docs = load.load()
    all_docs += docs

  # Load certif data
  loader = CSVLoader(file_path="/content/drive/MyDrive/LLM-project-piste-2/final-v1/certifications.csv")
  certif_docs = loader.load()

  return all_docs , certif_docs

# This function returns 2 questions related to the general idea of the chunk
def get_supplement(chunk, llm):
  template = """À partir du contenu du texte donné, génère uniquement deux questions générales en français qui couvrent la majorité du sens du texte. 
  Fournis uniquement les deux questions sans aucun texte supplémentaire. La réponse doit être impérativement en français. 
  La réponse doit respecter ce format:
  <question1>
  <question2>
  Texte : {text}
  """
  prompt = ChatPromptTemplate.from_template(template)
  prompt_with_context = prompt.format(text=chunk)
  supplement = llm.invoke(prompt_with_context).content

  return supplement

# This function takes a list of documents as an input and transform it to a markdown format, split it and then add the supplement questions for each chunk. It returns the list of all augmented chunks for all documents.
def get_docs_chunks(docs, llm, embedding_function):

  markdown_content = ""
  doc_chunks = []

  for doc in docs: 
    markdown_content = markdownify(doc.page_content).strip()
    text_splitter = MarkdownTextSplitter(embedding_function, breakpoint_threshold_type="interquartile")
    chunks = text_splitter.split_text(markdown_content)

    for i, chunk in enumerate(chunks):
      augmented_chunk = get_supplement(chunk, llm) + "\n" + chunk
      doc_ = Document(page_content=augmented_chunk, metadata=doc.metadata)
      doc_chunks.append(doc_)

  return doc_chunks
    

# Fonction pour enregistrer dans une base Chroma
def save_to_chroma(llm, embedding_function):

    all_docs, certif_docs = load_documents()
    all_docs_chunks = get_docs_chunks(all_docs, llm)

  # # Vector store for all websites
    CHROMA_DB_PATH = "/content/drive/MyDrive/LLM-project-piste-2/final-v1/chroma_db"
    
    db_all = Chroma.from_documents(all_docs_chunks, embedding_function, persist_directory=CHROMA_DB_PATH)
  
    # Vector store for formations websites
    CHROMA_CERTIF_PATH = "/content/drive/MyDrive/LLM-project-piste-2/final-v1/chroma_certif"
    db_certif = Chroma.from_documents(certif_docs, embedding_function, persist_directory=CHROMA_CERTIF_PATH)
  

def main():

    GROQ_API_KEY="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"

    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    # LLM to augment chunks
    llm_chunk = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # # Sauvegarder dans Chroma
    save_to_chroma(llm_chunk, embedding_function)
    
    # print("Documents chargés et indexés avec succès dans Chroma.")

if __name__ == "__main__":
    main()

