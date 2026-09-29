from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from dxa_qc import __version__
from dxa_qc.api import router
from dxa_qc.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title="DXA QC Lab", version=__version__)

    @app.middleware("http")
    async def disable_static_cache(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    app.include_router(router)
    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")
    return app


app = create_app()
