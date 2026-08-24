from langchain_core.prompts import ChatPromptTemplate

# Classification
classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You classify a text in a single business category."),
    ("human", "Text: The neural network analyzes X-rays."),
    ("ai", "Digital Health"),
    ("human", "Text: Kubernetes automates deployment."),
    ("ai", "Cloud"),
    ("human", "Text: {input}")
])

# Summary
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You summarize a text while preserving essential ideas."),
    ("human", "Text: {input}")
])

# Translation
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You translate from English to Spanish with a natural style. Respond only with a JSON object with a single field \"translated_text\" containing the translation."),
    ("human", "Text: {input}")
])