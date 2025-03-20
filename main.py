from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import Chunking

app = FastAPI()

class TextRequest(BaseModel):
    sentences: List[str]
    sim_threshold: Optional[float]
    overlap_threshold: Optional[float]
    check_subtitle_maxlength: Optional[int]
    min_chunk_tokens: Optional[int]
    max_chunk_tokens: Optional[int]
    compare_sentence_size: Optional[int]

@app.post("/chunk")
async def chunk(request: TextRequest):
    try:
        result = Chunking.generate_chunks(
            sentences=request.sentences
            , max_chunk_tokens=request.max_chunk_tokens
            , sim_threshold=request.sim_threshold
            , overlap_threshold=request.overlap_threshold
            , min_chunk_tokens=request.min_chunk_tokens
            , check_subtitle_maxlength=request.check_subtitle_maxlength
            , compare_sentence_size=request.compare_sentence_size
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))