"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

# Load .env manually if python-dotenv is not available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).strip().lower()
    return val in ("true", "1", "yes", "on")


def get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


# Application
APP_NAME = get_env("APP_NAME", "drug-discovery-lit-mining")
APP_VERSION = get_env("APP_VERSION", "1.0.0")
DEBUG = get_bool("DEBUG", False)

# Storage
DATA_DIR = Path(get_env("DDS_DATA_DIR", "./data")).resolve()
DATABASE_PATH = DATA_DIR / "app.db"
DATABASE_URL = get_env("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")

# Ollama (local LLM only - no cloud inference)
OLLAMA_HOST = get_env("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = get_env("OLLAMA_MODEL", "qwen2.5-1.5b-instruct")

# Inference
GPU_ENABLED = get_bool("GPU_ENABLED", False)

# Limits
MAX_PAPERS_PER_INVESTIGATION = get_int("MAX_PAPERS_PER_INVESTIGATION", 100)

# Literature acquisition
PUBMED_BASE_URL = get_env("PUBMED_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
PUBMED_EMAIL = get_env("PUBMED_EMAIL", "dds-local@example.com")
PUBMED_API_KEY = get_env("PUBMED_API_KEY", "")
PUBMED_RATE_LIMIT = get_int("PUBMED_RATE_LIMIT", 3)

EUROPEPMC_BASE_URL = get_env("EUROPEPMC_BASE_URL", "https://www.ebi.ac.uk/europepmc/webservices/rest/")

FULL_TEXT_DIR = Path(get_env("FULL_TEXT_DIR", "./data/full_text")).resolve()
MAX_IMPORT_FILE_SIZE_MB = get_int("MAX_IMPORT_FILE_SIZE_MB", 50)
ALLOWED_IMPORT_EXTENSIONS = get_env("ALLOWED_IMPORT_EXTENSIONS", ".pdf,.txt")

# NLP configuration
SPACY_MODEL = get_env("SPACY_MODEL", "en_core_web_sm")
SCISPACY_CHEMICAL_DISEASE_MODEL = get_env("SCISPACY_CHEMICAL_DISEASE_MODEL", "en_ner_bc5cdr_md")
SCISPACY_GENE_PROTEIN_MODEL = get_env("SCISPACY_GENE_PROTEIN_MODEL", "en_ner_bionlp13cg_md")
CHUNK_SIZE = get_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 50)
MAX_CHUNKS_PER_PAPER = get_int("MAX_CHUNKS_PER_PAPER", 200)
ENTITY_CONFIDENCE_THRESHOLD = get_float("ENTITY_CONFIDENCE_THRESHOLD", 0.5)
RELATIONSHIP_CONFIDENCE_THRESHOLD = get_float("RELATIONSHIP_CONFIDENCE_THRESHOLD", 0.4)

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FULL_TEXT_DIR.mkdir(parents=True, exist_ok=True)
