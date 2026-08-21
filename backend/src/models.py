from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class Scenario(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class AnalyzeRequest(BaseModel):
    scenario: Scenario
    question: str


class Strength(str, Enum):
    strong = "strong"
    moderate = "moderate"
    weak = "weak"


class Citation(BaseModel):
    source_id: str
    source_short_name: Optional[str] = None
    article_number: Optional[int] = None
    recital_number: Optional[int] = None
    annex_number: Optional[int] = None
    section: Optional[str] = None
    provision: Optional[str] = None
    quote: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one_target(self):
        targets = [
            self.article_number is not None,
            self.recital_number is not None,
            self.annex_number is not None,
        ]
        if sum(targets) != 1:
            raise ValueError(
                "A Citation must target exactly one of article_number, recital_number, or annex_number"
            )
        return self


class Finding(BaseModel):
    statement: str
    strength: Strength = Strength.moderate
    citations: List[Citation] = Field(default_factory=list)


class Answer(BaseModel):
    findings: List[Finding]
    actions: List[str]
    citations: List[Citation]


class Trace(BaseModel):
    workflow: str
    summary: str
    unsupported_claims_discarded: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    answer: Answer
    trace: Trace
    detailed_trace: Optional[List[dict]] = Field(default_factory=list)
    known_limitations: List[str] = Field(default_factory=list)
