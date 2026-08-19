from typing import List, Optional
from pydantic import BaseModel

class Scenario(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None

class AnalyzeRequest(BaseModel):
    scenario: Scenario
    question: str

class Citation(BaseModel):
    source_id: str
    source_short_name: Optional[str] = None
    article_number: Optional[int] = None
    quote: Optional[str] = None

class Finding(BaseModel):
    statement: str
    confidence: Optional[str] = None
    citations: List[Citation] = []

class Answer(BaseModel):
    findings: List[Finding]
    actions: List[str]
    citations: List[Citation]
    trace: dict
    detailed_trace: Optional[List[dict]] = []

class AnalyzeResponse(BaseModel):
    answer: Answer
