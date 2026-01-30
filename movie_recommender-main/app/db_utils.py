"""
Database utilities for PostgreSQL with pgvector.
Handles database connections and vector similarity search for movies.
"""

from typing import Dict, List

import psycopg2
from loguru import logger


class DatabaseManager:
    """Manager for database operations."""

    def __init__(self, host: str, port: str, database: str, user: str, password: str):
        """
        Initialize database manager.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

    def get_connection(self):
        """
        Get a database connection.

        Returns:
            psycopg2 connection object
        """
        return psycopg2.connect(**self.connection_params)

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar movies using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of movie dictionaries with metadata
        """
        logger.debug(f"Searching for {top_k} similar movies")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Use cosine distance for similarity search
            query = """
                SELECT
                    title,
                    overview,
                    genres,
                    tagline,
                    vote_average,
                    release_date,
                    runtime,
                    combined_text
                FROM movies
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            cursor.execute(query, (query_embedding, top_k))
            results = cursor.fetchall()
            logger.debug(f"Database returned {len(results)} results")

            # Convert to list of dictionaries
            movies = []
            for row in results:
                movies.append(
                    {
                        "title": row[0],
                        "overview": row[1],
                        "genres": row[2],
                        "tagline": row[3],
                        "vote_average": row[4],
                        "release_date": row[5],
                        "runtime": row[6],
                        "combined_text": row[7],
                    }
                )

            return movies

        finally:
            cursor.close()
            conn.close()

    def insert_movie(self, movie_data: Dict, embedding: List[float]) -> int:
        """
        Insert a movie with its embedding.

        Args:
            movie_data: Dictionary containing movie metadata
            embedding: Embedding vector

        Returns:
            ID of inserted movie
        """
        logger.debug(f"Inserting movie: {movie_data.get('title')}")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            query = """
                INSERT INTO movies (
                    movie_id, title, overview, genres, tagline,
                    vote_average, release_date, runtime,
                    combined_text, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            cursor.execute(
                query,
                (
                    movie_data.get("movie_id"),
                    movie_data.get("title"),
                    movie_data.get("overview"),
                    movie_data.get("genres"),
                    movie_data.get("tagline"),
                    movie_data.get("vote_average"),
                    movie_data.get("release_date"),
                    movie_data.get("runtime"),
                    movie_data.get("combined_text"),
                    embedding,
                ),
            )

            movie_id = cursor.fetchone()[0]
            conn.commit()

            return movie_id

        finally:
            cursor.close()
            conn.close()

    def count_movies(self) -> int:
        """
        Count total movies in database.

        Returns:
            Number of movies
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM movies")
            count = cursor.fetchone()[0]
            return count

        finally:
            cursor.close()
            conn.close()

    def clear_movies(self) -> None:
        """
        Clear all movies from database.
        """
        logger.info("Clearing all movies from database")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM movies")
            conn.commit()
            logger.info("All movies cleared from database")

        finally:
            cursor.close()
            conn.close()
