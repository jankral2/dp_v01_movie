"""
Data ingestion script for loading movie metadata CSV and creating embeddings.
This script loads movies from CSV, generates embeddings, and stores them in PostgreSQL.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional

import psycopg2
from sentence_transformers import SentenceTransformer


def parse_json_field(field_value: str) -> str:
    """
    Parse JSON field from CSV and extract names.

    Args:
        field_value: JSON string like '[{"id": 28, "name": "Action"}]'

    Returns:
        Comma-separated string of names, e.g., "Action, Adventure"
    """
    if not field_value or field_value == '[]':
        return ''

    try:
        items = json.loads(field_value)
        if isinstance(items, list):
            names = [item.get('name', '') for item in items if isinstance(item, dict)]
            return ', '.join(filter(None, names))
    except (json.JSONDecodeError, TypeError):
        pass

    return ''


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

    if movie.get('title'):
        parts.append(f"Title: {movie['title']}")

    if movie.get('genres'):
        parts.append(f"Genres: {movie['genres']}")

    if movie.get('overview'):
        parts.append(f"Plot: {movie['overview']}")

    if movie.get('tagline'):
        parts.append(f"Tagline: {movie['tagline']}")

    return '\n'.join(parts)


def load_movies_from_csv(csv_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load movies from CSV file.

    Args:
        csv_path: Path to movies_metadata.csv
        limit: Optional limit on number of movies to load

    Returns:
        List of movie dictionaries
    """
    movies = []

    print(f"Reading CSV from {csv_path}...")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            try:
                # Skip adult content
                if row.get('adult') == 'True':
                    continue

                # Parse genres from JSON
                genres = parse_json_field(row.get('genres', ''))

                # Extract relevant fields
                movie = {
                    'movie_id': row.get('id', ''),
                    'title': row.get('title', '').strip(),
                    'overview': row.get('overview', '').strip(),
                    'genres': genres,
                    'tagline': row.get('tagline', '').strip(),
                    'vote_average': row.get('vote_average', None),
                    'release_date': row.get('release_date', '').strip(),
                    'runtime': row.get('runtime', None)
                }

                # Skip if missing essential fields
                if not movie['title'] or not movie['overview']:
                    continue

                # Convert numeric fields
                try:
                    if movie['vote_average']:
                        movie['vote_average'] = float(movie['vote_average'])
                    if movie['runtime']:
                        movie['runtime'] = int(float(movie['runtime']))
                except (ValueError, TypeError):
                    pass

                movies.append(movie)

            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue

    print(f"Loaded {len(movies)} valid movies from CSV")
    return movies


def connect_to_db():
    """Connect to PostgreSQL database."""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'postgres'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres')
    )
    return conn


def ingest_movies(csv_path: str, limit: Optional[int] = None):
    """
    Main ingestion function.

    Args:
        csv_path: Path to movies_metadata.csv
        limit: Optional limit on number of movies to ingest
    """
    print("Starting movie data ingestion...")

    # Load embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load movies from CSV
    movies = load_movies_from_csv(csv_path, limit=limit)

    if not movies:
        print("No valid movies found in CSV!")
        return

    # Connect to database
    print("Connecting to database...")
    conn = connect_to_db()
    cursor = conn.cursor()

    # Clear existing data
    print("Clearing existing movie data...")
    cursor.execute("DELETE FROM movies")
    conn.commit()

    # Process movies
    print(f"\nProcessing {len(movies)} movies...")
    successful = 0

    for i, movie in enumerate(movies):
        try:
            # Create combined text for embedding
            combined_text = create_embedding_text(movie)

            # Generate embedding
            embedding = model.encode(combined_text).tolist()

            # Insert into database
            cursor.execute(
                """
                INSERT INTO movies (
                    movie_id, title, overview, genres, tagline,
                    vote_average, release_date, runtime,
                    combined_text, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (movie_id) DO NOTHING
                """,
                (
                    movie['movie_id'],
                    movie['title'],
                    movie['overview'],
                    movie['genres'],
                    movie['tagline'],
                    movie['vote_average'],
                    movie['release_date'],
                    movie['runtime'],
                    combined_text,
                    embedding
                )
            )

            successful += 1

            # Commit in batches
            if (i + 1) % 100 == 0:
                conn.commit()
                print(f"  Processed {i + 1}/{len(movies)} movies...")

        except Exception as e:
            print(f"Error processing movie '{movie.get('title')}': {e}")
            continue

    # Final commit
    conn.commit()
    cursor.close()
    conn.close()

    print(f"\nData ingestion complete!")
    print(f"Successfully stored {successful} movies in database")


if __name__ == "__main__":
    # Get CSV path from command line
    if len(sys.argv) < 2:
        print("Usage: python ingest_data.py <path_to_movies_metadata.csv> [limit]")
        print("Example: python ingest_data.py /data/movies_metadata.csv 1000")
        sys.exit(1)

    csv_file = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not Path(csv_file).exists():
        print(f"Error: CSV file not found at {csv_file}")
        sys.exit(1)

    ingest_movies(csv_file, limit=limit)
