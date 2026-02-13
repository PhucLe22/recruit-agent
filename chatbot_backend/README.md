# AI Agent Backend

A FastAPI-based backend service for AI-powered recruitment and CV processing.

## Features

- **User Management**: Create and manage user profiles
- **AI Matching**: Intelligent applicant-job matching using AI services
- **CV Processing**: Resume parsing, OCR, keyword extraction
- **Personality Assessment**: MBTI, Big Five, DISC scoring with Gemini AI
- **Job Suggestions**: AI-powered job matching based on resume analysis
- **LinkedIn Scraping**: Automated job data collection via Playwright
- **Google Integration**: Google Meet scheduling and calendar management

## Project Structure

```
chatbot_backend/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/v1/
│   │   ├── api.py                 # API router aggregation
│   │   └── endpoints/
│   │       ├── health.py          # Health check endpoints
│   │       ├── users.py           # User management
│   │       └── ai_matching.py     # AI matching endpoints
│   ├── core/
│   │   ├── config.py              # Centralized settings (Pydantic)
│   │   ├── database.py            # Async MongoDB connection
│   │   ├── dependencies.py        # FastAPI dependency injection
│   │   └── exceptions.py          # Global error handlers
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── cv_refinement/
│   │   │   ├── keyword_extraction.py
│   │   │   ├── improvement_suggestion.py
│   │   │   └── jobs_suggestion.py
│   │   ├── ingestion/
│   │   │   ├── pipeline.py        # Main resume processing pipeline
│   │   │   ├── pdf_processing.py
│   │   │   ├── ocr.py
│   │   │   └── text_preprocessing.py
│   │   ├── personality_assessment/
│   │   │   ├── scoring.py
│   │   │   └── job_recommendations.py
│   │   └── linkedin_webscraping/
│   │       └── webscraping.py
│   └── utils/
│       └── llm_utils.py           # LLM abstraction (Ollama/Gemini/HF)
├── data/
│   └── skills.yaml                # Skill categories and matching weights
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── requirements.txt               # All dependencies
├── requirements-base.txt          # Core dependencies
├── requirements-ml.txt            # ML/AI dependencies
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── run.py                         # Uvicorn entry point
```

## Quick Start

### Local Development

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Run
python run.py
```

### Docker

```bash
cp .env.example .env
docker compose up --build
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health/` | Health check |
| GET | `/api/v1/health/root` | API info |
| GET | `/api/v1/users/` | List users |
| POST | `/api/v1/users/` | Create user |
| GET | `/api/v1/users/{username}` | Get user |
| GET | `/api/v1/ai/applicant-matching/{job_id}` | Match applicants to job |
| GET | `/api/v1/ai/applicant-matching/all/{business_id}` | Match all for business |
| GET | `/api/v1/ai/applicant-profile/{user_id}` | Applicant AI profile |

## Testing

```bash
pytest tests/ -v
```

## Environment Variables

See [.env.example](.env.example) for all available configuration options.

Key variables:
- `MONGO_ATLAS_URI` - MongoDB connection string
- `AI_SERVICE_URL` - External Node.js AI service URL
- `DEFAULT_MODEL` - Default LLM model (e.g. `qwen2.5:3b`)
- `GEMINI_API_KEY` - Google Gemini API key
- `CORS_ORIGINS` - Allowed CORS origins (JSON array)
