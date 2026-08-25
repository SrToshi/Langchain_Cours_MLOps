from src.core.chains import classification_chain, summary_chain, translation_chain

text = """
Surrounded by countryside, this farmhouse offers peace and quiet in the grounds of a stately home. We think you will  spend most of your time in the garden, complete with a wood-fired hot tub and dining table for meals alfresco. Head inside, and the ground floor is arranged for easy socialising and relaxing, while upstairs, the bedrooms ensure an uninterrupted night’s sleep. If exploring is on the agenda, Stratford-upon-Avon is a twenty-minute drive away, or why not stretch your legs with a walk through the Cotswolds? Reached in less than thirty minutes by car.
"""

# Classification
print("\n--- Classification ---")
classification = classification_chain.invoke({"input": text})
print("Category:", classification.category)
print("Confidence:", classification.confidence)

# Summary
print("\n--- Summary ---")
summary = summary_chain.invoke({"input": text})
print("Summary:", summary.summary)

# Translation
print("\n--- Translation ---")
translation = translation_chain.invoke({"input": text})
print("Translated Text:", translation.translated_text)
