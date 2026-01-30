"""
Data Management Page - Upload and manage movie data.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from db_utils import DatabaseManager
from embedding_service import EmbeddingService
from ingest_service import ingest_movies
from loguru import logger

# Configure loguru (if not already configured)
if not logger._core.handlers:
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )


# Page config
st.set_page_config(page_title="Data Management", page_icon="📊", layout="wide")

st.title("📊 Data Management")
st.markdown("Upload and manage movie data in the database.")


# Initialize services
@st.cache_resource
def init_services():
    """Initialize embedding service and database manager."""
    logger.info("Initializing services for Data Management page")
    embedding_service = EmbeddingService()
    db_manager = DatabaseManager(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )
    logger.info("Services initialized successfully")
    return embedding_service, db_manager


try:
    embedding_service, db_manager = init_services()
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()


# Display current database stats
st.markdown("---")
st.markdown("### Current Database Status")

col1, col2 = st.columns(2)

with col1:
    try:
        movie_count = db_manager.count_movies()
        st.metric("Movies in Database", f"{movie_count:,}")
    except Exception as e:
        st.error(f"Error getting movie count: {e}")

with col2:
    st.info("Embeddings are generated locally using all-MiniLM-L6-v2")


# Data ingestion section
st.markdown("---")
st.markdown("### Import Movie Data")

# Find CSV files in /data directory
data_dir = Path("/data")
csv_files = []
if data_dir.exists():
    csv_files = [f.name for f in data_dir.glob("*.csv")]

if not csv_files:
    st.warning(
        "No CSV files found in /data directory. Please add movies_metadata.csv to the data folder."
    )
    st.stop()

# File selection
selected_file = st.selectbox(
    "Select CSV file", csv_files, index=0 if "movies_metadata.csv" in csv_files else 0
)

# Configuration
col1, col2 = st.columns(2)

with col1:
    limit = st.number_input(
        "Number of movies to import (0 = all)",
        min_value=0,
        max_value=50000,
        value=1000,
        step=100,
        help="Start with 1000 for testing. Set to 0 to import all movies.",
    )

with col2:
    clear_existing = st.checkbox(
        "Clear existing data",
        value=True,
        help="Remove all existing movies before importing",
    )

# Start ingestion button
if st.button("🚀 Start Import", type="primary", use_container_width=True):
    csv_path = data_dir / selected_file
    logger.info(
        f"Starting import from {csv_path} (limit: {limit}, clear_existing: {clear_existing})"
    )

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()

    # Progress callback
    def update_progress(current: int, total: int, message: str):
        if total > 0:
            progress = current / total
            progress_bar.progress(min(progress, 1.0))
        status_text.text(message)

    try:
        # Run ingestion
        stats = ingest_movies(
            csv_path=str(csv_path),
            db_manager=db_manager,
            embedding_service=embedding_service,
            limit=limit if limit > 0 else None,
            clear_existing=clear_existing,
            progress_callback=update_progress,
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Import completed in {elapsed_time:.1f}s - Stats: {stats}")

        # Clear progress
        progress_bar.empty()
        status_text.empty()

        # Display results
        st.success(f"✅ Import completed in {elapsed_time:.1f} seconds!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Processed", stats["total"])
        with col2:
            st.metric(
                "Successfully Imported",
                stats["successful"],
                delta=None if stats["errors"] == 0 else f"-{stats['errors']} errors",
            )
        with col3:
            st.metric("Errors", stats["errors"], delta_color="inverse")

        # Refresh button
        if st.button("🔄 Refresh Stats"):
            st.rerun()

    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during import: {e}")
        st.exception(e)


# Instructions
st.markdown("---")
st.markdown("### Instructions")
st.markdown("""
1. **Place CSV file**: Add `movies_metadata.csv` to the `/data` directory
2. **Select file**: Choose the CSV file from the dropdown
3. **Configure import**:
   - Set number of movies to import (1000 recommended for testing)
   - Choose whether to clear existing data
4. **Start import**: Click the button to begin importing
5. **Wait**: The process will show progress and may take a few minutes

**Note**: Generating embeddings is done locally and is relatively fast.
A typical import of 1000 movies takes about 1-2 minutes.
""")
