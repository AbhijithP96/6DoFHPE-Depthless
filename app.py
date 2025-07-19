from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "6DoFHPE with Depth Estimation Model"}


@app.post("/process_frame/")
async def process_frame():
    pass