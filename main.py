from typing import Union
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get values from environment variables
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 3007))

app = FastAPI()


@app.get("/", response_class=PlainTextResponse)
def read_root():
    return "Reporting from Oracle Service!"


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run("main:app", port=3007, log_level="info")
