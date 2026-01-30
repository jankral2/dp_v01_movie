# Movie recommender - RAG Application

A simple Retrieval Augmented Generation (RAG) application for thesis demonstration. This system uses vector similarity search with PostgreSQL pgvector and Google Gemini API to answer questions based on a document database.

## Architecture

The application consists of two main containers orchestrated by Docker Compose:

1. **PostgreSQL with pgvector** - Database for storing documents and their embeddings
2. **Streamlit App** - Web-based chat interface for user interaction

### Technology Stack

- **Database**: PostgreSQL 17 with pgvector extension
- **Embeddings**: all-MiniLM-L6-v2 (384-dimensional vectors, running locally)
- **LLM**: Google Gemini API (gemini-1.5-flash or gemini-2.0-flash)
- **Frontend**: Streamlit
- **Language**: Python 3.12
- **Package Management**: pip with uv-generated requirements

## Project Structure

```
Movie recommender/
├── app/
│   ├── streamlit_app.py          # Main Streamlit chat interface
│   ├── embedding_service.py      # Embedding generation service
│   ├── db_utils.py               # Database utilities
│   ├── requirements.in           # Direct dependencies
│   ├── requirements.txt          # Locked dependencies
│   └── Dockerfile                # Streamlit app container
├── scripts/
│   ├── db-init/
│   │   └── 01-create-vector.sql  # Database initialization
│   └── ingest_data.py            # Data loading script
├── data/                          # Text files for ingestion
├── docker-compose.yaml            # Container orchestration
├── .env.template                  # Environment variables template
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for containers
- Google Gemini API key (get one free at https://aistudio.google.com/apikey)
- Internet connection for API calls

### First Time Setup

1. **Clone the repository** (or navigate to project directory)
   ```bash
   cd /path/to/Movie recommender
   ```

2. **Create environment file and add your Google Gemini credentials**
   ```bash
   cp .env.template .env
   ```
   Edit the `.env` file and add your Google Gemini credentials:
   ```
   GOOGLE_API_KEY=your-api-key
   GOOGLE_MODEL_NAME=gemini-2.0-flash
   ```

3. **Prepare data files** (optional)
   ```bash
   # Create data directory if it doesn't exist
   mkdir -p data
   ```

   Download [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv] and add .csv file into data/ directory


4. **Start the services**
   ```bash
   docker compose up --build
   ```

   This will:
   - Start PostgreSQL and create the `text_data` table
   - Build and start the Streamlit application with local embedding model

5. **Ingest data into the database**
   ```bash
   # Run the data ingestion script
   docker exec movierec-app python /app/scripts/ingest_data.py /data
   ```

6. **Access the application**
   Open your browser and navigate to: [http://localhost:8501](http://localhost:8501)

### Normal Usage

After the first setup, you can start the application with:

```bash
docker compose up
```

To stop the application:
```bash
docker compose down
```

To stop and remove all data (including database):
```bash
docker compose down -v
```

## Using the Application

1. **Access the web interface** at [http://localhost:8501](http://localhost:8501)
2. **Enter your question** in the text input field
3. **View retrieved documents** in the expandable section
4. **Read the AI-generated answer** based on your documents

### Configuration

You can adjust settings in the sidebar:
- **Number of documents to retrieve**: Controls how many relevant chunks are used for context (1-10)

## Data Ingestion

The data ingestion script processes text files from the `data/` directory is done in streamlit app.

**How it works:**
1. Loads all `.txt` files from the specified directory
2. Chunks each document into ~500 character segments with 50 character overlap
3. Generates embeddings for each chunk using all-MiniLM-L6-v2
4. Stores chunks and embeddings in PostgreSQL

**Note**: Re-running the ingestion script will clear existing data and reload everything.

## Development

### Project Dependencies

To update dependencies:

1. Edit `app/requirements.in` to add/remove packages
2. Generate new lockfile:
   ```bash
   cd app
   uv pip compile requirements.in -o requirements.txt
   ```
3. Rebuild the container:
   ```bash
   docker compose up --build
   ```

### Database Access

To access the PostgreSQL database directly:

```bash
docker exec -it movierec-db psql -U postgres -d movierec
```

Useful queries:
```sql
-- Count documents
SELECT COUNT(*) FROM text_data;

-- View sample documents
SELECT id, LEFT(text, 100) FROM text_data LIMIT 5;

-- Test similarity search
SELECT text FROM text_data
WHERE embedding IS NOT NULL
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

## Troubleshooting

### Google Gemini API errors
If you see API authentication errors:
- Verify your `GOOGLE_API_KEY` in the `.env` file
- Check that your model name is valid (e.g., `gemini-2.0-flash`, `gemini-1.5-flash`)
- Verify your API key is valid and has not expired
- Check your quota at https://aistudio.google.com/apikey

### Database connection errors
Ensure the database is healthy:
```bash
docker compose ps
docker compose logs db
```

### Rate limiting errors
If you encounter rate limiting from Google Gemini:
- Wait a few seconds between requests
- Check your Google AI Studio quota limits
- The free tier has generous limits for testing


## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `db` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `postgres` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `postgres` | Database password |
| `GOOGLE_API_KEY` | (required) | Google Gemini API key |
| `GOOGLE_MODEL_NAME` | (required) | Google Gemini model name (e.g., `gemini-2.0-flash`) |

## License

This is a thesis project for educational purposes.
