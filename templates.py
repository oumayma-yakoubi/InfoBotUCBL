def get_qa_template():
  qa_template = """
    Réponds à la question en te basant uniquement sur le contexte fourni. Si aucun contexte n'est fourni, réponds d'une manière naturelle.

    Reformule toujours la réponse pour qu'elle soit claire et concise, tout en restant pertinente par rapport au contexte donné.

    Si vous ne disposez pas d'une information, dites simplement que vous ne l'avez pas, sans mentionner de contexte ou de recherche infructueuse."

    Avec ce type de directive, le modèle répondra directement, par exemple :
    "Je ne dispose pas de cette information."

    Si tu ne peux pas répondre à la question en te basant sur le contexte, réponds :
    
    'Je vous invite à consulter [lien] pour obtenir plus d'informations.'

    Ajoute systématiquement une phrase invitant l'utilisateur à consulter le lien fourni, en terminant la réponse comme suit :
    Pour plus d'informations, veuillez consulter [lien].


    Contexte: {context}

    Question: {question}
    """
  return qa_template

def get_cv_template():
    cv_template = """Génère un modèle de lettre de motivation en français pour la formation demandée par l'utilisateur en te basant sur le contexte qui va suivre.

    Rédige un en-tête pour une lettre formelle.

    Fais attention à remplacer le destinataire avec l'information du responsable de la formation. 
    
    L'adresse devra aussi être remplacée par l'adresse physique du département de la formation.

    Commence par un paragraphe sur le parcours et les compétences de l'utilisateur s'il les a renseignés dans la question.
    Si l'utilisateur n'a pas renseigné d'informations à propos de lui-même dans la question, essaie de mettre des propositions entre crochets dans la lettre. 

    Continue avec un autre paragraphe sur l'établissement et la formation voulue.
    Ne mentionne PAS d'aspects trop précis de la formation comme les unités d'enseignements. 
    Dis à l'utilisateur entre parenthèses à la fin du paragraphe d'aller se renseigner sur les unités d'enseignement de la formation lui-même pour compléter le paragraphe avec des exemples précis.

    Finis la lettre avec un paragraphe où l'utilisateur exprime sa motivation pour intégrer la formation.   

    Signe la lettre.

    À la toute fin de ta réponse, donne l'adresse mail de la scolarité et le site de la formation.
    Rappelle à l'utilisateur de bien se renseigner sur la formation, son contenu et ses débouchés afin d'être sûr qu'il veut bien l'intégrer.

    Contexte : {context}

    Question : {question}
    """

    return cv_template

def get_recommendation_template():

  # certif_template = """
  #   Contexte des UEs extraites : {context_formation}
  #   Contexte des certificats recommandés : {context_certificats}
  #   Question : {question}

  #   Utilise les informations ci-dessus pour fournir une recommandation détaillée de certificats à l'utilisateur. Explique clairement :
  #   - Le nom du certificat
  #   - Les compétences acquises
  #   - Une justification de la correspondance entre les UEs et les certificats recommandés.

  #   """

  certif_template = """
    Contexte des UEs suivies dans la formation : {context_formation}
    Contexte des certificats recommandés : {context_certificats}

    En te basant sur les informations ci-dessus, rédige une recommandation détaillée de certificats à obtenir. Assure-toi d'expliquer :
      Les compétences clés développées par chaque certificat
    montre clairement comment ces certificat renforcent et complètent la formation en question.

    Adopte un ton professionnel et pédagogique pour aider l'utilisateur à comprendre la valeur ajoutée de ces certificats dans son parcours académique et professionnel. Sois clair, structuré et cohérent dans ta réponse.
    """

  return certif_template