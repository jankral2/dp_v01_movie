-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create movies table for storing movie metadata and embeddings
CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    movie_id TEXT UNIQUE,           -- from CSV 'id' column
    title TEXT NOT NULL,
    overview TEXT,
    genres TEXT,                    -- e.g., "Action, Sci-Fi, Thriller"
    tagline TEXT,
    vote_average NUMERIC,           -- for showing ratings
    release_date TEXT,              -- movie release date
    runtime INTEGER,                -- runtime in minutes
    combined_text TEXT,             -- full text used for embedding
    embedding vector(384),          -- all-MiniLM-L6-v2 produces 384-dimensional vectors
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS movies_embedding_idx ON movies
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
