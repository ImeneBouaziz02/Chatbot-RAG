
import os
import psycopg
from pgvector.psycopg import register_vector
import ollama
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Charger les variables d'environnement
load_dotenv('../src/.env')

print("✅ Importations réussies!")


# Chaîne de connexion
db_connection_str = f"dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')} host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')}"

conn_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Modèle d'embedding Ollama (768 dimensions)
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# Modèle de génération
LLM_MODEL = "llama3.2"

print("✅ Variables configurées!")

try:
    conn = psycopg.connect(**conn_params)
    register_vector(conn)
    print("✅ Connexion à Supabase réussie!")
    
    # Vérifier pgvector
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        result = cur.fetchone()
        if result:
            print("✅ Extension pgvector est active!")
    
    conn.close()
except Exception as e:
    print(f"❌ Erreur de connexion: {e}")

# Vérifier qu'Ollama fonctionne
try:
    ollama.list()
    print("✅ Ollama est opérationnel!")
except Exception as e:
    print(f"❌ Ollama n'est pas démarré. Lancez-le puis réessayez.")

def create_conversation_list(url: str = None, file_path: str = None) -> list[str]:
    """
    Télécharge et extrait le corpus depuis une URL ou un fichier local
    """
    if url:
        response = requests.get(url)
        response.encoding = 'utf-8'
        text = response.text
    elif file_path:
        with open(file_path, "r", encoding='utf-8') as file:
            text = file.read()
    else:
        raise ValueError("Vous devez fournir soit une URL, soit un file_path")
    
    soup = BeautifulSoup(text, 'html.parser')
    
    corpus_list = []
    for p in soup.find_all('p'):
        text = p.get_text().strip()
        if len(text) > 50 and not text.startswith("<"):
            text = text.removeprefix("     ")
            corpus_list.append(text)
    
    print(f"✅ {len(corpus_list)} documents extraits")
    return corpus_list

# Télécharger le corpus
conversation_url = "https://www.info.univ-tours.fr/~antoine/parole_publique/Accueil_UBS/index.html"
corpus_list = create_conversation_list(url=conversation_url)
print(f"Premier document: {corpus_list[0][:100]}...")


def calculate_embeddings(corpus: str) -> list[float]:
    """
    Génère un embedding avec Ollama (local)
    """
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=corpus
    )
    return response['embedding']

# Test
test_text = "Ceci est un test pour vérifier les embeddings."
test_embedding = calculate_embeddings(test_text)
print(f"✅ Embedding généré! Dimension: {len(test_embedding)}")
print(f"Premiers valeurs: {test_embedding[:5]}")

def save_embedding(corpus: str, embedding: list[float], cursor) -> None:
    """
    Insère un document et son embedding dans la base
    """
    cursor.execute(
        '''INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)''',
        (corpus, embedding)
    )

print("✅ Fonction save_embedding définie!")
with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    register_vector(conn)
    
    with conn.cursor() as cur:
        cur.execute("""DROP TABLE IF EXISTS embeddings""")
        print("🗑️ Table supprimée si elle existait")
        
        cur.execute("""CREATE EXTENSION IF NOT EXISTS vector""")
        print("✅ Extension vector activée")
        
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY, 
                corpus TEXT,
                embedding vector({EMBEDDING_DIM})
            );
        """)
        print("✅ Table 'embeddings' créée avec pgvector")
        
        # Insérer les documents
        sample_size = min(20, len(corpus_list))
        print(f"\n📥 Insertion de {sample_size} documents...")
        
        for i, corpus in enumerate(corpus_list[:sample_size]):
            print(f"Traitement {i+1}/{sample_size}...", end='\r')
            embedding = calculate_embeddings(corpus=corpus)
            save_embedding(corpus=corpus, embedding=embedding, cursor=cur)
        
        conn.commit()
        print(f"\n✅ {sample_size} documents insérés avec succès!")

def similar_corpus(input_corpus: str, db_connection_str: str, top_k: int = 3) -> list[tuple]:
    """
    Trouve les documents les plus similaires à l'entrée
    """
    query_embedding = calculate_embeddings(input_corpus)
    
    with psycopg.connect(db_connection_str) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, corpus, 1 - (embedding <=> %s::vector) as similarity
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = cur.fetchall()
    
    return results

# Test de recherche
query = "Quelle est l'importance de l'éducation?"
results = similar_corpus(query, db_connection_str, top_k=3)

print(f"🔍 Résultats pour: '{query}'\n")
for id, text, similarity in results:
    print(f"ID: {id} | Similarité: {similarity:.4f}")
    print(f"Texte: {text[:150]}...\n")

def generate_response(query: str, db_connection_str: str) -> str:
    """
    Génère une réponse en utilisant l'architecture RAG avec Ollama
    """
    # 1. Récupérer les documents pertinents
    relevant_docs = similar_corpus(query, db_connection_str, top_k=3)
    
    # 2. Construire le contexte
    context = "\n\n".join([doc[1] for doc in relevant_docs])
    
    # 3. Créer le prompt
    prompt = f"""Contexte tiré de documents sur la parole publique:
{context}

Question: {query}

Instructions:
- Réponds à la question en te basant UNIQUEMENT sur le contexte fourni
- Si l'information n'est pas dans le contexte, dis clairement "Je ne trouve pas cette information dans les documents fournis"
- Sois concis et précis
- Réponds en français

Réponse:"""
    
    # 4. Générer la réponse avec Ollama
    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt
    )
    
    return response['response']

# Test de génération (output_1.txt)
# question = "Quels sont les principaux thèmes abordés dans les discours?"
# print(f"❓ Question: {question}\n")

# reponse = generate_response(question, db_connection_str)
# print(f"🤖 Réponse:\n{reponse}")

def chatbot_loop():
    """
    Boucle interactive pour tester le chatbot
    """
    print("\n" + "="*60)
    print("🤖 CHATBOT RAG avec Ollama - Tapez 'quit' pour quitter")
    print("="*60 + "\n")
    
    while True:
        user_input = input("Vous: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Au revoir!")
            break
        
        if not user_input.strip():
            continue
        
        try:
            response = generate_response(user_input, db_connection_str)
            print(f"\n🤖 Chatbot: {response}\n")
        except Exception as e:
            print(f"❌ Erreur: {e}\n")

# Décommenter pour lancer le chatbot interactif (output_2.txt)
chatbot_loop()


with psycopg.connect(db_connection_str) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM embeddings")
        count = cur.fetchone()[0]
        
        cur.execute("SELECT id, corpus FROM embeddings LIMIT 1")
        sample = cur.fetchone()
        
        print("📊 Statistiques de la base de données:")
        print(f"  - Nombre de documents: {count}")
        print(f"  - Dimension des embeddings: {EMBEDDING_DIM}")
        print(f"  - Modèle d'embedding: {EMBEDDING_MODEL}")
        print(f"  - Modèle de génération: {LLM_MODEL}")
        print(f"  - Exemple: {sample[1][:100]}...")

print("\n✅ Notebook terminé avec succès avec Ollama!")