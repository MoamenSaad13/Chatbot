# chatbot_api.py - Enhanced FastAPI Chatbot Service

import pandas as pd
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import re
import os
from typing import List, Optional, Dict, Any
from beanie import Document, init_beanie
import motor.motor_asyncio
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import uvicorn
import pymongo
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load environment variables ---
# load_dotenv()
load_dotenv(dotenv_path="C:\\Users\\Moamen\\Desktop\\Chatbot\\.env")

# --- Application Settings ---
DATABASE_URI = os.getenv("DATABASE_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "test")

# --- Model Settings ---
GEMINI_MODEL_NAME = "models/gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_THRESHOLD = 0.7
TRACK_NAME_MATCH_THRESHOLD = 0.85

# --- Global variables ---
df_tracks: Optional[pd.DataFrame] = None
official_track_names: List[str] = []
keyword_index: Optional[faiss.Index] = None
track_name_index: Optional[faiss.Index] = None
embedder_instance: Optional[SentenceTransformer] = None
gemini_model: Optional[genai.GenerativeModel] = None


# --- MongoDB Models ---
class Roadmap(Document):
    title: str
    requirments: str  # Note: keeping the typo from original schema
    target_audience: str

    class Settings:
        name = "roadmaps"


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model_config = ConfigDict(from_attributes=True)


class ChatSession(Document):
    session_id: str = Field(..., unique=True)
    messages: List[ChatMessage] = Field(default_factory=list)
    last_suggested_roadmap: Optional[str] = None
    roadmap_confirmed: bool = False
    rejected_roadmaps: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = "chat_sessions"


# --- API Request/Response Models ---
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the chat session")
    user_input: str = Field(..., min_length=1, description="User's message")


class ChatResponse(BaseModel):
    assistant_message: str = Field(..., description="AI assistant's response")
    session_id: str = Field(..., description="Session identifier")
    suggested_track: Optional[str] = Field(None, description="Suggested learning track if any")
    tracks_available: List[str] = Field(default_factory=list, description="Available tracks")


class SessionInfoResponse(BaseModel):
    session_id: str
    message_count: int
    last_suggested_roadmap: Optional[str]
    roadmap_confirmed: bool
    rejected_roadmaps: List[str]
    created_at: datetime
    updated_at: datetime


class TracksResponse(BaseModel):
    tracks: List[str]
    total_count: int


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
    database_connected: bool
    models_loaded: bool


# --- Helper Functions ---
async def load_roadmap_data_from_mongodb():
    """Load roadmap data from MongoDB and build FAISS indices"""
    global df_tracks, official_track_names, keyword_index, track_name_index, embedder_instance

    try:
        logger.info("Loading roadmap data from MongoDB...")
        roadmaps = await Roadmap.find_all().to_list()

        if not roadmaps:
            logger.warning("No roadmap data found in MongoDB")
            df_tracks = pd.DataFrame(columns=["track", "keywords", "matching interests"])
            official_track_names = []
            return

        roadmap_data = []
        for roadmap in roadmaps:
            track_name = roadmap.title
            keywords = roadmap.requirments if roadmap.requirments else ""
            interests = roadmap.target_audience if roadmap.target_audience else ""
            roadmap_data.append([track_name, keywords, interests])

        df_tracks = pd.DataFrame(roadmap_data, columns=["track", "keywords", "matching interests"])
        official_track_names = df_tracks["track"].astype(str).tolist()
        keywords_list = df_tracks["keywords"].astype(str).tolist()
        matching_interests = df_tracks["matching interests"].astype(str).tolist()

        logger.info(f"Found {len(official_track_names)} tracks from MongoDB")

        # Build keyword/interest FAISS index
        all_texts_to_embed = keywords_list + matching_interests
        if any(t for t in all_texts_to_embed if t):
            logger.info("Generating embeddings for keyword/interest FAISS index...")
            valid_texts = [t for t in all_texts_to_embed if t]
            if valid_texts:
                keyword_embeddings = embedder_instance.encode(valid_texts, normalize_embeddings=True)
                if keyword_embeddings.size > 0:
                    keyword_index = faiss.IndexFlatIP(keyword_embeddings.shape[1])
                    keyword_index.add(np.array(keyword_embeddings).astype("float32"))
                    logger.info("Keyword/Interest FAISS index built successfully")

        # Build track name FAISS index
        logger.info("Generating embeddings for track name FAISS index...")
        valid_track_names = [t for t in official_track_names if t]
        if valid_track_names:
            track_name_embeddings = embedder_instance.encode(valid_track_names, normalize_embeddings=True)
            if track_name_embeddings.size > 0:
                track_name_index = faiss.IndexFlatIP(track_name_embeddings.shape[1])
                track_name_index.add(np.array(track_name_embeddings).astype("float32"))
                logger.info("Track name FAISS index built successfully")

    except Exception as e:
        logger.error(f"Error loading roadmap data: {e}")
        raise


def get_relevant_tracks_from_keywords(user_message: str) -> List[str]:
    """Find relevant tracks based on user message keywords"""
    if keyword_index is None or embedder_instance is None or not official_track_names:
        return []

    try:
        query_vec = embedder_instance.encode([user_message], normalize_embeddings=True)
        D, I = keyword_index.search(np.array(query_vec).astype("float32"), k=min(3, keyword_index.ntotal))
        results = []

        if I.ndim == 2 and I.size > 0:
            for i in range(I.shape[1]):
                score = D[0, i]
                index_hit = I[0, i]
                if score >= FAISS_THRESHOLD and index_hit >= 0:
                    track_idx = index_hit % len(official_track_names)
                    if 0 <= track_idx < len(official_track_names):
                        results.append(official_track_names[track_idx])

        return list(set(results))
    except Exception as e:
        logger.error(f"Error during keyword FAISS search: {e}")
        return []


def find_closest_official_track(suggested_track_name: str, threshold: float) -> Optional[str]:
    """Find the closest official track name using FAISS similarity search"""
    if track_name_index is None or embedder_instance is None or not suggested_track_name or not official_track_names:
        return None

    try:
        query_vec = embedder_instance.encode([suggested_track_name], normalize_embeddings=True)
        query_vec_float32 = np.array(query_vec).astype("float32")
        D, I = track_name_index.search(query_vec_float32, k=1)

        if I.ndim == 2 and I.size > 0 and D.ndim == 2 and D.size > 0:
            matched_index = I[0, 0]
            score = D[0, 0]
            if matched_index >= 0 and score >= threshold:
                if 0 <= matched_index < len(official_track_names):
                    return official_track_names[matched_index]
        return None
    except Exception as e:
        logger.error(f"Error during track name FAISS search for '{suggested_track_name}': {e}")
        return None


def extract_suggested_track(assistant_message: str) -> Optional[str]:
    """Extract suggested track name from assistant's message"""
    # Look for bold text first
    match = re.search(r'\*\*(.*?)\*\*', assistant_message)
    if match:
        track_name = match.group(1).strip(" .:,!?")
        if len(track_name) > 3 and "track" not in track_name.lower() and "path" not in track_name.lower():
            return track_name

    # Look for "recommend the X track" pattern
    match = re.search(r'recommend the\s+(.*?)\s+track', assistant_message, re.IGNORECASE)
    if match:
        return match.group(1).strip(" .:,!?")

    return None


def is_off_topic(user_input: str) -> bool:
    """Check if user input is off-topic"""
    off_topic_keywords = ["كلمه عبيطه"]

    learning_keywords = [
        "learn", "teach", "course", "track", "skill", "programming", "develop", "code",
        "study", "career", "tech", "data", "web", "mobile", "AI", "cloud", "security",
        "frontend", "backend", "fullstack", "devops", "cybersecurity", "blockchain",
        "game dev", "embedded", "iot", "ui/ux", "qa", "testing", "engineer", "analyst",
        "scientist", "developer", "path", "roadmap", "guide", "advice", "recommend",
        "tutorial", "lesson", "education", "training", "certification"
    ]

    user_input_lower = user_input.lower()
    has_learning_keyword = any(keyword in user_input_lower for keyword in learning_keywords)
    has_off_topic_keyword = any(keyword in user_input_lower for keyword in off_topic_keywords)

    if has_learning_keyword:
        return False
    if has_off_topic_keyword and not has_learning_keyword:
        return True

    return False


# --- System Prompt Template ---
SYSTEM_PROMPT_TEMPLATE = """
You are a professional and emotionally intelligent AI assistant guiding users to find the most suitable programming learning track *exclusively from our platform's database*. Your goal is to conduct a personalized, adaptive conversation — especially for users who may not even know what they're interested in yet.

**User Profile Context (use this to guide your conversation and recommendations):**
- User's experience level: {experience_level}
- User's technical interests: {technical_interests}
- User's personal goals: {personal_goals}
- Tracks user has previously rejected (DO NOT suggest these again): {rejected_tracks}

**Domain Restriction:** You MUST only engage with questions related to learning programming, technology skills, or educational tracks. If the user asks something unrelated (e.g., weather, recipes, jokes, sports scores), politely respond ONLY with: "I'm here to help you choose the best learning track. Unfortunately, I can't assist with this topic." Do not elaborate further.

**Conversation Flow & Behavior Rules:**
- Begin with warm, friendly energy. Start with light, friendly, indirect questions to discover the user's personality and preferences. For example:
    - "What kind of things do you enjoy doing in your free time?"
    - "Do you like solving problems, designing visuals, or organizing information?"
- If the user seems uncertain (e.g., replies like "I don't know" or "anything"), guide them gently with **multiple-choice questions** such as:
    - "Would you say you're more creative, analytical, or practical?"
    - "Are you more interested in building websites, mobile apps, or working with data?"

- **You MUST explore at least 3 dimensions before recommending any track.** Dimensions can include:
    1. Learning style (e.g., "Do you prefer videos, reading, or hands-on practice?")
    2. Learning speed (e.g., "Do you like to learn quickly, or take your time exploring?")
    3. Personal goal (e.g., "Do you want to build a portfolio, get a job, or explore for fun?")
  These should be phrased conversationally and naturally woven into the dialogue.

- When discussing creative interests like "designing visuals", DO NOT default only to UI/UX. Instead, offer a **diverse set of creative-tech options**, such as:
    - 🎮 Game Development
    - 🌐 Web Animation
    - 🎨 Creative Coding
    - 📱 Interactive Mobile Apps

- If a user **rejects a track**, respond with empathy and curiosity. You MUST ask a gentle follow-up like:
    - "Got it! Just to help me improve suggestions — was it too design-heavy, too technical, or something else?"
    - "علشان أقدر أرشح أفضل، ممكن أعرف إيه اللي مكنش مناسب في المسار ده؟" (Arabic)
    - "为了给您提供更好的建议，能告诉我这个课程有什么不适合您的地方吗？" (Chinese)

- When you identify a suitable track, **bold the track name** like this: "Based on your interests, I think the **Front-End Development** track would be perfect for you."

- After suggesting a track, **always ask if they'd like to know more** about it or if they'd prefer a different suggestion.

- If the user confirms interest in a track, provide a brief, enthusiastic summary of what they'll learn and potential career outcomes.

**Track Recommendation Guidelines:**
- Recommend tracks that align with the user's experience level, interests, and goals.
- For beginners with no clear preference, suggest accessible entry points like Front-End Development or UI/UX Design.
- For users with analytical interests, consider Data Science, Back-End Development, or AI tracks.
- For creative users, consider UI/UX Design, Front-End, Game Development, or Mobile App tracks.
- For users interested in infrastructure or systems, consider DevOps, Cloud Computing, or Cybersecurity.

**Multilingual Support:**
- If a user communicates in a language other than English, respond in that same language.
- Maintain the same conversation quality and recommendation approach regardless of language.

Remember, your goal is to make the user feel understood and guide them to a track they'll be excited about, even if they initially have no idea what they want to learn.
"""

# --- FastAPI App ---
app = FastAPI(
    title="Chatbot API Service",
    description="AI-powered chatbot for learning track recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Initialization ---
async def initialize_chatbot_dependencies():
    """Initialize all chatbot dependencies"""
    global embedder_instance, gemini_model

    logger.info("Initializing chatbot dependencies...")

    try:
        # Validate environment variables
        if not DATABASE_URI or not GEMINI_API_KEY:
            raise ValueError("DATABASE_URI and GEMINI_API_KEY must be set in .env file")

        # Initialize MongoDB connection
        client = motor.motor_asyncio.AsyncIOMotorClient(DATABASE_URI)
        db = client[MONGO_DATABASE_NAME]
        await init_beanie(database=db, document_models=[ChatSession, Roadmap])
        logger.info(f"Connected to MongoDB database '{MONGO_DATABASE_NAME}'")

        # Initialize Gemini AI
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info("Gemini AI model initialized")

        # Initialize sentence transformer
        embedder_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Sentence transformer model loaded")

        # Load roadmap data and build indices
        await load_roadmap_data_from_mongodb()

        logger.info("Chatbot dependencies initialized successfully")

    except Exception as e:
        logger.error(f"CRITICAL ERROR during chatbot initialization: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    await initialize_chatbot_dependencies()


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down chatbot service...")


# --- API Endpoints ---

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for processing user messages"""
    session_id = request.session_id
    user_input = request.user_input.strip()

    try:
        # Get or create chat session
        chat_session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not chat_session:
            chat_session = ChatSession(session_id=session_id)
            await chat_session.create()
            logger.info(f"Created new chat session: {session_id}")

        # Check if input is off-topic
        if is_off_topic(user_input):
            assistant_response_message = "I'm here to help you choose the best learning track. Unfortunately, I can't assist with this topic."
            suggested_track = None
        else:
            # Process user input
            user_input_lower = user_input.lower()
            rejection_keywords = ["no", "not interested", "don't like", "something else", "different"]
            acceptance_keywords = ["yes", "interested", "sounds good", "tell me more", "like it"]

            # Handle track acceptance/rejection
            if chat_session.last_suggested_roadmap:
                if any(keyword in user_input_lower for keyword in rejection_keywords):
                    if chat_session.last_suggested_roadmap not in chat_session.rejected_roadmaps:
                        chat_session.rejected_roadmaps.append(chat_session.last_suggested_roadmap)
                        logger.info(f"User rejected track: {chat_session.last_suggested_roadmap}")
                    chat_session.last_suggested_roadmap = None
                    await chat_session.save()
                elif any(keyword in user_input_lower for keyword in acceptance_keywords):
                    chat_session.roadmap_confirmed = True
                    logger.info(f"User confirmed track: {chat_session.last_suggested_roadmap}")
                    await chat_session.save()

            # Find relevant tracks
            relevant_tracks = get_relevant_tracks_from_keywords(user_input)
            relevant_tracks = [track for track in relevant_tracks if track not in chat_session.rejected_roadmaps]

            # Prepare context for AI
            context = f"""
            Experience Level: Not specified
            Technical Interests: Not specified
            Personal Goals: Not specified
            Rejected Roadmaps: {', '.join(chat_session.rejected_roadmaps) if chat_session.rejected_roadmaps else 'None'}
            Relevant Tracks Found: {', '.join(relevant_tracks) if relevant_tracks else 'None'}
            """

            # Get conversation history (last 10 messages)
            conversation_history = [f"{msg.role}: {msg.content}" for msg in chat_session.messages[-10:]]

            # Generate system prompt
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                experience_level="Not specified",
                technical_interests="Not specified",
                personal_goals="Not specified",
                rejected_tracks=", ".join(chat_session.rejected_roadmaps) if chat_session.rejected_roadmaps else "None"
            )

            # Create full prompt
            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nConversation History:\n" + "\n".join(
                conversation_history) + f"\n\nUser: {user_input}\n\nAssistant:"

            # Generate AI response
            response = gemini_model.generate_content(full_prompt)
            assistant_response_message = response.text

            # Extract and validate suggested track
            suggested_track = extract_suggested_track(assistant_response_message)
            if suggested_track:
                official_track = find_closest_official_track(suggested_track, TRACK_NAME_MATCH_THRESHOLD)
                if official_track:
                    chat_session.last_suggested_roadmap = official_track
                    suggested_track = official_track
                    await chat_session.save()
                    logger.info(f"Suggested track: {official_track}")

        # Save messages to session
        chat_session.messages.append(ChatMessage(role="user", content=user_input))
        chat_session.messages.append(ChatMessage(role="assistant", content=assistant_response_message))
        chat_session.updated_at = datetime.now()
        await chat_session.save()

        return ChatResponse(
            assistant_message=assistant_response_message,
            session_id=session_id,
            suggested_track=suggested_track,
            tracks_available=official_track_names
        )

    except Exception as e:
        logger.error(f"Error in chat handler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while processing your message"
        )


@app.get("/api/v1/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """Get information about a specific chat session"""
    try:
        chat_session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        return SessionInfoResponse(
            session_id=chat_session.session_id,
            message_count=len(chat_session.messages),
            last_suggested_roadmap=chat_session.last_suggested_roadmap,
            roadmap_confirmed=chat_session.roadmap_confirmed,
            rejected_roadmaps=chat_session.rejected_roadmaps,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving session information"
        )


@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        chat_session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        await chat_session.delete()
        logger.info(f"Deleted session: {session_id}")

        return {"message": "Session deleted successfully", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting session"
        )


@app.get("/api/v1/tracks", response_model=TracksResponse)
async def get_available_tracks():
    """Get all available learning tracks"""
    try:
        return TracksResponse(
            tracks=official_track_names,
            total_count=len(official_track_names)
        )
    except Exception as e:
        logger.error(f"Error getting tracks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving available tracks"
        )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_connected = False
        models_loaded = False

        try:
            # Test database connection
            await Roadmap.find_one()
            db_connected = True
        except:
            pass

        # Check if models are loaded
        if embedder_instance is not None and gemini_model is not None:
            models_loaded = True

        return HealthResponse(
            status="healthy" if db_connected and models_loaded else "degraded",
            service="chatbot-api",
            timestamp=datetime.now(),
            database_connected=db_connected,
            models_loaded=models_loaded
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            service="chatbot-api",
            timestamp=datetime.now(),
            database_connected=False,
            models_loaded=False
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Chatbot API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# --- Main ---
if __name__ == "__main__":
    uvicorn.run(
        "chatbot_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )