"""
Movie data ingestion service for loading CSV and creating embeddings.
"""

import csv
import json
from typing import Callable, Dict, List, Optional

from loguru import logger


def parse_json_field(field_value: str) -> str:
    """
    Parse JSON field from CSV and extract names.

    Args:
        field_value: JSON string like '[{"id": 28, "name": "Action"}]'

    Returns:
        Comma-separated string of names
    """
    if not field_value or field_value == "[]":
        return ""

    try:
        items = json.loads(field_value)
        if isinstance(items, list):
            names = [item.get("name", "") for item in items if isinstance(item, dict)]
            return ", ".join(filter(None, names))
    except (json.JSONDecodeError, TypeError):
        pass

    return ""


def create_embedding_text(movie: Dict) -> str:
    """
    Create combined text for embedding from movie data.

    Format:
    Title: {title}
    Genres: {genres}
    Plot: {overview}
    Tagline: {tagline}

    Args:
        movie: Dictionary containing movie metadata

    Returns:
        Combined text string
    """
    parts = []

    if movie.get("title"):
        parts.append(f"Title: {movie['title']}")

    if movie.get("genres"):
        parts.append(f"Genres: {movie['genres']}")

    if movie.get("overview"):
        parts.append(f"Plot: {movie['overview']}")

    if movie.get("tagline"):
        parts.append(f"Tagline: {movie['tagline']}")

    return "\n".join(parts)


def load_movies_from_csv(
    csv_path: str,
    limit: Optional[int] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> List[Dict]:
    """
    Load movies from CSV file.

    Args:
        csv_path: Path to movies_metadata.csv
        limit: Optional limit on number of movies to load
        progress_callback: Optional callback(count, message) for progress updates

    Returns:
        List of movie dictionaries
    """
    logger.info(f"Loading movies from CSV: {csv_path} (limit: {limit or 'none'})")
    movies = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            try:
                # Skip adult content
                if row.get("adult") == "True":
                    continue

                # Parse genres from JSON
                genres = parse_json_field(row.get("genres", ""))

                # Extract relevant fields
                movie = {
                    "movie_id": row.get("id", ""),
                    "title": row.get("title", "").strip(),
                    "overview": row.get("overview", "").strip(),
                    "genres": genres,
                    "tagline": row.get("tagline", "").strip(),
                    "vote_average": row.get("vote_average", None),
                    "release_date": row.get("release_date", "").strip(),
                    "runtime": row.get("runtime", None),
                }

                # Skip if missing essential fields
                if not movie["title"] or not movie["overview"]:
                    continue

                # Convert numeric fields
                try:
                    if movie["vote_average"]:
                        movie["vote_average"] = float(movie["vote_average"])
                    if movie["runtime"]:
                        movie["runtime"] = int(float(movie["runtime"]))
                except (ValueError, TypeError):
                    pass

                movies.append(movie)

                # Progress callback every 100 rows
                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback(
                        len(movies), f"Loaded {len(movies)} valid movies..."
                    )

            except Exception as e:
                logger.warning(f"Error processing row {i}: {e}")
                continue

    logger.info(f"Loaded {len(movies)} valid movies from CSV")
    return movies


def ingest_movies(
    csv_path: str,
    db_manager,
    embedding_service,
    limit: Optional[int] = None,
    clear_existing: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, int]:
    """
    Ingest movies from CSV into database.

    Args:
        csv_path: Path to movies_metadata.csv
        db_manager: DatabaseManager instance
        embedding_service: EmbeddingService instance
        limit: Optional limit on number of movies
        clear_existing: Whether to clear existing data
        progress_callback: Optional callback(current, total, message) for progress

    Returns:
        Dictionary with statistics (total, successful, errors)
    """
    logger.info(f"Starting movie ingestion from {csv_path}")

    # Load movies from CSV
    if progress_callback:
        progress_callback(0, 0, "Loading CSV file...")

    movies = load_movies_from_csv(
        csv_path,
        limit=limit,
        progress_callback=lambda count, msg: progress_callback(count, count, msg)
        if progress_callback
        else None,
    )

    if not movies:
        logger.warning("No movies loaded from CSV")
        return {"total": 0, "successful": 0, "errors": 0}

    # Clear existing data if requested
    if clear_existing:
        logger.info("Clearing existing movies from database")
        if progress_callback:
            progress_callback(0, len(movies), "Clearing existing movies...")
        db_manager.clear_movies()
        logger.info("Existing movies cleared")

    # Process movies
    successful = 0
    errors = 0

    for i, movie in enumerate(movies):
        try:
            # Create combined text for embedding
            combined_text = create_embedding_text(movie)

            # Generate embedding
            embedding = embedding_service.encode(combined_text)

            # Insert into database
            movie["combined_text"] = combined_text
            db_manager.insert_movie(movie, embedding)

            successful += 1

            # Progress callback
            if progress_callback:
                progress_callback(
                    i + 1, len(movies), f"Processing: {movie['title'][:50]}..."
                )

        except Exception as e:
            logger.error(f"Error processing movie '{movie.get('title')}': {e}")
            errors += 1
            continue

    logger.info(
        f"Ingestion complete: {successful} successful, {errors} errors out of {len(movies)} total"
    )

    return {"total": len(movies), "successful": successful, "errors": errors}
