"""
Regula — Regulatory Research & Compliance Assistant
Backend entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Regula",
    description="Regulatory research and compliance assistant",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/analyze")
async def analyze(query: dict):
    """Placeholder for analysis workflow"""
    return {"message": "Analysis workflow not yet implemented"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
