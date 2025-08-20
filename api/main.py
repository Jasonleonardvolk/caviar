from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="TORI API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "api_port": int(os.environ.get("API_PORT", 0))}

class Echo(BaseModel):
    text: str

@app.post("/api/echo")
def api_echo(payload: Echo):
    return {"echo": payload.text}

if __name__ == "__main__":
    # For ad-hoc local runs (not used by launcher)
    import uvicorn
    port = int(os.environ.get("API_PORT", "8002"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
