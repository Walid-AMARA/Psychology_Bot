from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import google.generativeai as genai

app = FastAPI()

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation contexts (use a database for production)
conversations = {}

# Initialize the Gemini client with your API key
genai.configure(api_key="KEY HERE")

class AnalyzeRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    conversation_id: str
    message: str

@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    psychological_data = request.text

    if not psychological_data.strip():
        return {"error": "Le texte fourni est vide."}

    # Generate a unique conversation ID
    conversation_id = str(uuid.uuid4())
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Prepare the initial prompt
    # Make sure to deal with special characters in the frontend
    prompt = (
        "Vous etes un assistant de resources humaines specialiser dans laliniement entre l'afiche du poste et la personaliter du candidtat"
        "tu doit deviner les competences (soft skills) requies d'un job meme si ils ne sont pas mentioner dans la fiche du poste"
        "Je vais te donner un résultat d’un test de personnalité."
        "Je voudrais que tu me répondes par rapport à la pertinence du profil pour différents jobs. "
        "Je veux que tu répondes d’une façon nuancée en donnant les raisons pourquoi la personne est un bon fit pour le job "
        "et en donnant aussi les axes d’amélioration qu’il faut travailler. "
        # "répondez sous forme de puces, chaque note ou observation ou pensée a sa propre ligne."
        "Je veut que si l'utilisateur te pose des question qui n'ont pas de rapport avec le job fit que tu rependre que c'est hors de ton scope"
        # "Vos réponses ne seront pas affichées sur le site officiel de Google Gemini, il est donc essentiel que vous répondiez dans un format qui me permette d'utiliser votre réponse comme réponse d'API, N'utilisez pas de langage Markdown, n'utilisez pas d'astérisques ni de caractères spéciaux"
        f"Voici les données : {psychological_data}"
    )

    # Generate the initial response
    response = model.generate_content(prompt)

    if not response or not response.text:
        return {"error": "La génération du contenu a échoué."}

    # Start a new chat session with the generated response
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": prompt},
            {"role": "model", "parts": response.text},
        ]
    )

    # Store the chat object for further interactions
    conversations[conversation_id] = chat

    return {"conversation_id": conversation_id, "analysis": response.text}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    conversation_id = request.conversation_id
    message = request.message.strip()

    if conversation_id not in conversations:
        return {"error": "ID de conversation invalide."}

    if not message:
        return {"error": "Le message est vide."}

    # Retrieve the chat object for this conversation
    chat = conversations[conversation_id]

    # Send the user's message and generate a response
    try:
        response = chat.send_message(message)
        if not response or not response.text:
            return {"error": "La réponse du modèle est vide."}

        # Update the conversation history (handled internally by the chat object)
        return {"reply": response.text}
    except Exception as e:
        return {"error": f"Une erreur est survenue: {str(e)}"}
