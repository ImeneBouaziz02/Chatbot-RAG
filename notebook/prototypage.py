#faire les importations nécessaires


#Déclarer les variables nécessaires
conversation_file_path = "..."
#initialiser le client OpenAI en cas d'utiliser la bibliothèque openai ou autre bibliothèque compatible
openai_client= OpenAI(api_key="")

db_connection_str="dbname=..  user= ..  password=.. host=.. port=.."


def create_conversation_list(file_path:str)->list[str]:
    with open(file_path, "r") as file:
        text = file.read()
        text_list = text.split("\n")
        filtered_list = [chaine.removeprefix("     ") for chaine in text_list if not chaine.startswith("<")]
        print(filtered_list)
        return filtered_list

def calculate_embeddings(corpus:str,client: OpenAI)->list[float]:
    embeddings=client.embeddings.create(input=corpus,model=model,encoding_format="float").data
    return embeddings[0].embedding

def save_embedding(corpus:str, embedding:list[float],cursor: Cursor)->None:
    cursor.execute('''INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)''', (corpus, embedding))


##définir une fonnction similar_corpus qui prend en entrée un texte et renvoie les textes similaires dans la base de données
#..à compléter
#def similar_corpus(input_corpus:str, client:OpenAI, db_connection_str:str)->tuple[int, str,list[float]]:


with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""DROP TABLE IF EXISTS embeddings""")

        #creer l'extension pgvector
        #....à compléter



        cur.execute("""CREATE TABLE IF NOT EXISTS embeddings (ID SERIAL PRIMARY KEY,
                    corpus TEXT,
                    embedding FLOAT8[]);  # pour intégrer pgvector, remplacer FLOAT8[] par VECTOR(1536) , 1536 étant la dimension des embeddings qui dépend du modèle utilisé
                    """)
        corpus_list=create_conversation_list(file_path=conversation_file_path)

        for corpus in corpus_list:
            embedding = calculate_embeddings(corpus=corpus, client=openai_client)
            save_embedding(corpus=corpus, embedding=embedding, cursor=cur)
        conn.commit()



#introduire une requete pour interroger
