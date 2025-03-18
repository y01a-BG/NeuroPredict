from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Get port from environment variable for Cloud Run compatibility
PORT = int(os.getenv("PORT", "8000"))

# Create FastAPI instance
app = FastAPI(
    title="Hello World API",
    description="A minimal FastAPI Hello World example",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def read_root():
    """
    Root endpoint that returns a Hello World message
    """
    return {"message": "Hello, World!"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    """
    Endpoint that says hello to a specific name
    """
    return {"message": f"Hello, {name}!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
