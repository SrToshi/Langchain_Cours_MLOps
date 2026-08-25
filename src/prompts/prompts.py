from langchain_core.prompts import ChatPromptTemplate

# Classification
classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You classify a text in a single business category. Respond only with a JSON object with two fields: \"category\" (string) and \"confidence\" (a float between 0 and 1)."),
    ("human", "Text: The neural network analyzes X-rays."),
    ("ai", '{{"category": "Digital Health", "confidence": 0.95}}'),
    ("human", "Text: Kubernetes automates deployment."),
    ("ai", '{{"category": "Cloud", "confidence": 0.95}}'),
    ("human", "Text: {input}")
])

# Summary
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You summarize a text while preserving essential ideas. Respond only with a JSON object with a single field \"summary\" containing the summary."),
    ("human", "Text: {input}")
])

# Translation
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You translate from English to Spanish with a natural style. Respond only with a JSON object with a single field \"translated_text\" containing the translation."),
    ("human", "Text: {input}")
])