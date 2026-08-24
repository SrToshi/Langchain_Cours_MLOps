from pydantic import BaseModel, Field

class ClassificationResult(BaseModel):
    category: str = Field(description="Category chosen for the text")
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence score between 0 and 1"
    )

class SummaryResult(BaseModel):
    summary: str = Field(description="Short and faithful summary of the text")

class TranslationResult(BaseModel):
    translated_text: str = Field(description="Text translated to English")