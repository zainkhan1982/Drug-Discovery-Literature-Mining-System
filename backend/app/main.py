"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.investigations import router as investigations_router
from app.api.literature import router as literature_router
from app.api.nlp import router as nlp_router
from app.config import APP_NAME, APP_VERSION, DEBUG
from app.database import DATABASE_PATH, init_database
from app.logging import logger
from app.schemas import HealthOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", APP_NAME, APP_VERSION)
    init_database()
    logger.info("Database ready")
    yield
    logger.info("Shutting down %s", APP_NAME)


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME, version=APP_VERSION, debug=DEBUG, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthOut, tags=["health"])
    def health() -> HealthOut:
        with get_connection_for_health() as conn:
            fk_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        return HealthOut(
            status="ok",
            version=APP_VERSION,
            database=str(DATABASE_PATH),
            foreign_keys=bool(fk_on),
        )

    app.include_router(investigations_router)
    app.include_router(literature_router)
    app.include_router(nlp_router)
    return app


def get_connection_for_health():
    from app.database import get_connection
    return get_connection()


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
