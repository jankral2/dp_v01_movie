"""
Movie Recommender - RAG-based Movie Recommendation Application
A Streamlit chat interface that uses RAG to recommend and discuss movies.
"""

import sys

import streamlit as st
from db_utils import DatabaseManager
from embedding_service import EmbeddingService
from google import genai
from loguru import logger
from pydantic_settings import BaseSettings

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    DB_HOST: str
    DB_PORT: str
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    GOOGLE_API_KEY: str
    GOOGLE_MODEL_NAME: str


@st.cache_resource
def init_settings() -> Settings:
    """Load and validate settings at startup."""
    logger.info("Loading settings from environment variables...")
    settings = Settings()
    logger.info(
        f"Settings loaded: DB={settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )
    return settings


# Initialize services
@st.cache_resource
def init_services(_settings: Settings):
    """Initialize embedding service and database manager."""
    logger.info("Initializing services...")
    embedding_service = EmbeddingService()
    logger.info("Embedding service initialized")

    db_manager = DatabaseManager(
        host=_settings.DB_HOST,
        port=_settings.DB_PORT,
        database=_settings.DB_NAME,
        user=_settings.DB_USER,
        password=_settings.DB_PASSWORD,
    )
    logger.info("Database manager initialized")
    return embedding_service, db_manager


@st.cache_resource
def init_llm_client(_settings: Settings):
    """Initialize Google Gemini client."""
    logger.info("Initializing Google Gemini client...")
    client = genai.Client(api_key=_settings.GOOGLE_API_KEY)
    logger.info(
        f"Google Gemini client initialized with model: {_settings.GOOGLE_MODEL_NAME}"
    )
    return client


def search_similar_movies(query: str, embedding_service, db_manager, top_k: int = 5):
    """
    Search for similar movies using vector similarity.

    Args:
        query: User's question or search query
        embedding_service: EmbeddingService instance
        db_manager: DatabaseManager instance
        top_k: Number of results to return

    Returns:
        List of movie dictionaries
    """
    logger.info(f"Searching for movies with query: '{query}' (top_k={top_k})")

    # Generate embedding for query
    query_embedding = embedding_service.encode(query)
    logger.debug(f"Generated embedding with dimension: {len(query_embedding)}")

    # Search database for similar movies
    results = db_manager.search_similar(query_embedding, top_k=top_k)
    logger.info(f"Found {len(results)} similar movies")

    return results


def format_movie_context(movies: list) -> str:
    """
    Format movies as context for LLM.

    Args:
        movies: List of movie dictionaries

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, movie in enumerate(movies, 1):
        parts = [f"[{i}] {movie['title']}"]

        if movie.get("release_date"):
            parts.append(f"({movie['release_date'][:4]})")

        if movie.get("genres"):
            parts.append(f"- Genres: {movie['genres']}")

        if movie.get("vote_average"):
            parts.append(f"- Rating: {movie['vote_average']:.1f}/10")

        if movie.get("runtime"):
            parts.append(f"- Runtime: {movie['runtime']} min")

        if movie.get("overview"):
            parts.append(f"- Plot: {movie['overview']}")

        if movie.get("tagline"):
            parts.append(f"- Tagline: {movie['tagline']}")

        context_parts.append("\n".join(parts))

    return "\n\n".join(context_parts)


def generate_response(query: str, movies: list, llm_client, settings: Settings):
    """
    Generate response using LLM with RAG context.

    Args:
        query: User's question
        movies: List of relevant movie dictionaries
        llm_client: Google Gemini GenerativeModel instance
        settings: Application settings

    Returns:
        Generated response
    """
    logger.info(f"Generating LLM response for query: '{query}'")

    # Prepare context from retrieved movies
    context_text = format_movie_context(movies)

    # Create prompt
    prompt = f"""You are a helpful movie recommendation assistant. Provide thoughtful recommendations and insights about movies based on the provided context.

    Based on the following movie recommendations, please answer the user's question.
    Provide helpful information about the movies, including why they might match what the user is looking for.
    If the user asks for recommendations, explain why these movies are relevant.

    Movies:
    {context_text}

    User Question: {query}

    Answer:"""

    logger.debug(f"Using model: {settings.GOOGLE_MODEL_NAME}")

    # Generate response using Google Gemini
    response = llm_client.models.generate_content(
        model=settings.GOOGLE_MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.7,
            "max_output_tokens": 700,
        },
    )

    logger.info("LLM response generated successfully")
    return response.text


def display_movie_card(movie: dict, index: int):
    """
    Display a movie as a card with metadata.

    Args:
        movie: Movie dictionary
        index: Index for numbering
    """
    with st.container():
        # Title with year
        title = movie["title"]
        if movie.get("release_date"):
            year = movie["release_date"][:4] if len(movie["release_date"]) >= 4 else ""
            if year:
                title = f"{title} ({year})"

        st.markdown(f"### {index}. {title}")

        # Metadata row
        metadata_parts = []
        if movie.get("vote_average"):
            rating = movie["vote_average"]
            stars = "⭐" * int(round(rating / 2))
            metadata_parts.append(f"{stars} {rating:.1f}/10")

        if movie.get("genres"):
            metadata_parts.append(movie["genres"])

        if movie.get("runtime"):
            metadata_parts.append(f"{movie['runtime']} min")

        if metadata_parts:
            st.markdown(f"**{'  •  '.join(metadata_parts)}**")

        # Tagline
        if movie.get("tagline"):
            st.markdown(f'*"{movie["tagline"]}"*')

        # Overview
        if movie.get("overview"):
            st.markdown(movie["overview"])

        st.markdown("---")


def main():
    """Main Streamlit application."""
    logger.info("Starting Movie Recommender application")

    st.set_page_config(
        page_title="Movie Recommender - RAG", page_icon="🎬", layout="wide"
    )

    st.title("🎬 Movie Recommender - RAG Application")
    st.markdown("Discover movies using AI-powered semantic search and recommendations.")

    # Initialize settings and services
    try:
        settings = init_settings()
        embedding_service, db_manager = init_services(settings)
        llm_client = init_llm_client(settings)
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        st.error(f"Error initializing services: {e}")
        st.stop()
        return

    # Constants
    DISPLAY_TOP_N = 5  # Number of movies to show in UI

    # Sidebar with settings
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider(
            "Number of movies to retrieve from database",
            min_value=5,
            max_value=50,
            value=10,
        )
        st.caption(f"Top {DISPLAY_TOP_N} most similar movies will be displayed")

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This is a RAG (Retrieval Augmented Generation) movie recommendation system."
        )
        st.markdown(f"**LLM:** {settings.GOOGLE_MODEL_NAME}")
        st.markdown("**Embeddings:** all-MiniLM-L6-v2")

        # Display movie count
        try:
            movie_count = db_manager.count_movies()
            st.markdown(f"**Movies in database:** {movie_count:,}")
        except Exception:
            pass

        st.markdown("---")
        st.markdown("### Example Queries")
        st.markdown("- *Action movies with time travel*")
        st.markdown("- *Romantic comedies set in New York*")
        st.markdown("- *Sci-fi movies about AI*")
        st.markdown("- *Movies similar to Inception*")

    # Main interface
    user_query = st.text_input(
        "🔍 What kind of movies are you looking for?",
        placeholder="e.g., 'thrilling heist movies' or 'heartwarming family films'",
    )

    if user_query:
        with st.spinner("Searching for movies..."):
            # Search for similar movies (retrieve top_k from database)
            similar_movies = search_similar_movies(
                user_query, embedding_service, db_manager, top_k=top_k
            )

            if not similar_movies:
                st.warning(
                    "No movies found in the database. Please ingest movie data first."
                )
                st.info(
                    "Run: `docker exec movierec-app python /app/scripts/ingest_data.py /data/movies_metadata.csv`"
                )
                st.stop()

        # Movies to display (top N by similarity)
        display_movies = similar_movies[:DISPLAY_TOP_N]

        # Generate AI response (using all retrieved movies for better context)
        with st.spinner("Generating recommendation summary..."):
            try:
                response = generate_response(
                    user_query, similar_movies, llm_client, settings
                )

                # Display response right after input
                st.markdown("### 💡 AI Recommendation")
                st.markdown(response)

            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.markdown("**Debug info:**")
                st.code(str(e))

        # Display retrieved movies (only top N)
        with st.expander(f"🎥 Top {len(display_movies)} Similar Movies", expanded=True):
            for i, movie in enumerate(display_movies, 1):
                display_movie_card(movie, i)


if __name__ == "__main__":
    main()
