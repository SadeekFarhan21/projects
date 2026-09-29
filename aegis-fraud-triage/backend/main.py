"""Entry point for Aegis backend"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
