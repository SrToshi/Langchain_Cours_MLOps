from src.core.llm import llm
from src.core.schemas import ClassificationResult
from src.core.schemas import TranslationResult
from src.core.schemas import SummaryResult
from src.prompts.prompts import classification_prompt
from src.prompts.prompts import translation_prompt
from src.prompts.prompts import summary_prompt

classification_chain = classification_prompt | llm.with_structured_output(ClassificationResult, method="json_mode")
translation_chain = translation_prompt | llm.with_structured_output(TranslationResult, method="json_mode")
summary_chain = summary_prompt | llm.with_structured_output(SummaryResult, method="json_mode")