import os
import psycopg
from pgvector.psycopg import register_vector
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Charger les variables d'environnement
load_dotenv('../src/.env')

# Configurer Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

print("✅ Importations réussies!")

# Chaîne de connexion
db_connection_str = f"dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')} host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')}"

# Paramètres de connexion (alternative)
conn_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Modèle d'embedding Gemini
EMBEDDING_MODEL = "models/embedding-001"
# Dimension des embeddings Gemini (768)
EMBEDDING_DIM = 768

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
        else:
            print("⚠️ Extension pgvector n'est pas installée. Exécutez: CREATE EXTENSION vector;")
    
    conn.close()
except Exception as e:
    print(f"❌ Erreur de connexion: {e}")


def create_conversation_list(url: str = None, file_path: str = None) -> list[str]:
    """
    Télécharge et extrait le corpus depuis une URL ou un fichier local
    """
    if url:
        # Télécharger depuis l'URL
        response = requests.get(url)
        response.encoding = 'utf-8'
        text = response.text
    elif file_path:
        # Lire depuis un fichier local
        with open(file_path, "r", encoding='utf-8') as file:
            text = file.read()
    else:
        raise ValueError("Vous devez fournir soit une URL, soit un file_path")
    
    # Parser avec BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    
    # Extraire les paragraphes
    corpus_list = []
    for p in soup.find_all('p'):
        text = p.get_text().strip()
        # Nettoyer et filtrer
        if len(text) > 50 and not text.startswith("<"):
            text = text.removeprefix("     ")
            corpus_list.append(text)
    
    print(f"✅ {len(corpus_list)} documents extraits")
    return corpus_list

# Test
conversation_url = "https://www.info.univ-tours.fr/~antoine/parole_publique/Accueil_UBS/index.html"
corpus_list = create_conversation_list(url=conversation_url)
print(f"Premier document: {corpus_list[0][:100]}...")


def calculate_embeddings(corpus: str) -> list[float]:
    """
    Génère un embedding avec Gemini
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=corpus,
        task_type="retrieval_document"
    )
    return result['embedding']

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

# Créer la table et insérer les embeddings
with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    register_vector(conn)
    
    with conn.cursor() as cur:
        # Supprimer la table si elle existe
        cur.execute("""DROP TABLE IF EXISTS embeddings""")
        print("🗑️ Table supprimée si elle existait")
        
        # Créer l'extension pgvector
        cur.execute("""CREATE EXTENSION IF NOT EXISTS vector""")
        print("✅ Extension vector activée")
        
        # Créer la table avec pgvector
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY, 
                corpus TEXT,
                embedding vector({EMBEDDING_DIM})
            );
        """)
        print("✅ Table 'embeddings' créée avec pgvector")
        
        # Insérer les documents (limiter pour le test)
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
    
    Returns:
        Liste de tuples (id, corpus, similarity)
    """
    # Générer l'embedding de la requête
    query_embedding = calculate_embeddings(input_corpus)
    
    # Rechercher dans la base
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
    Génère une réponse en utilisant l'architecture RAG
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
    
    # 4. Générer la réponse avec Gemini
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    
    return response.text

# Test de génération
question = "Quels sont les principaux thèmes abordés dans les discours?"
print(f"❓ Question: {question}\n")

reponse = generate_response(question, db_connection_str)
print(f"🤖 Réponse:\n{reponse}")

def chatbot_loop():
    """
    Boucle interactive pour tester le chatbot
    """
    print("\n" + "="*60)
    print("🤖 CHATBOT RAG - Tapez 'quit' pour quitter")
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

# Décommenter pour lancer le chatbot interactif
# chatbot_loop()

with psycopg.connect(db_connection_str) as conn:
    with conn.cursor() as cur:
        # Compter les documents
        cur.execute("SELECT COUNT(*) FROM embeddings")
        count = cur.fetchone()[0]
        
        # Exemple de document
        cur.execute("SELECT id, corpus FROM embeddings LIMIT 1")
        sample = cur.fetchone()
        
        print("📊 Statistiques de la base de données:")
        print(f"  - Nombre de documents: {count}")
        print(f"  - Dimension des embeddings: {EMBEDDING_DIM}")
        print(f"  - Exemple de document (ID {sample[0]}): {sample[1][:100]}...")

print("\n✅ Notebook terminé avec succès!")