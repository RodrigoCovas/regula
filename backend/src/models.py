from typing import List, Optional
from pydantic import BaseModel, Field

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
    section: Optional[str] = None
    provision: Optional[str] = None
    quote: Optional[str] = None

class Finding(BaseModel):
    statement: str
    confidence: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)

class Answer(BaseModel):
    findings: List[Finding]
    actions: List[str]
    citations: List[Citation]
    trace: dict
    detailed_trace: Optional[List[dict]] = Field(default_factory=list)

class AnalyzeResponse(BaseModel):
    answer: Answer
