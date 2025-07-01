---
title: "🔍 Preventing Duplicate Content with AI Embeddings: My Practical Approach"
date: 2025-07-01 08:00:00 +0200
lang: en
layout: post
categories: [AI, NLP]
tags: [openai, embeddings, content, automation]
---
In my latest project, I automate the collection and rewriting of daily news articles using AI. While AI rewriting works surprisingly well, I quickly ran into a challenge: **how do you prevent your website from publishing content that’s essentially the same, just worded differently?**

## 🧠 The Challenge: Semantic Duplicates

AI can easily paraphrase sentences. For example:

1.  "I eat McDonalds"
2. "I consume fast food"

These look different at first glance, but the core message is the same. Simple text comparison or keyword matching won’t catch this — and if you’re publishing daily content, you need to avoid filling your site with **semantically duplicated articles**.

## 🧬 The Solution: Embeddings

The solution lies in the use of **embeddings**.

An embedding is a mathematical vector (i.e. a list of numbers) that represents the semantic meaning of a sentence or document. These vectors are generated in a high-dimensional space. When two texts are similar in meaning, their vectors are "close" to each other in this space.

OpenAI offers an API that generates such embeddings.

## ⚙️ Generating Embeddings with OpenAI

Here’s a simple Python function to generate embeddings from text using the OpenAI API:

```python
import openai

openai.api_key = "your-api-key"

def get_embedding(text: str, model: str = "text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
```

This returns a list of roughly 1536 float values that represent the input text’s meaning.

## 🔁 Comparing Embeddings with Cosine Similarity

To detect semantic duplicates, I use cosine similarity — a common way to measure the angle between two vectors. A score close to 1.0 means high similarity.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def is_similar(embedding1, embedding2, threshold=0.85):
    similarity = cosine_similarity(
        [embedding1],
        [embedding2]
    )[0][0]
    return similarity >= threshold
```
**Example:**
Suppose you’ve already stored an embedding of a previous article. You can compare it to a new one like this:
```python
old_embedding = get_embedding("I eat McDonalds")
new_embedding = get_embedding("I consume fast food")

if is_similar(old_embedding, new_embedding):
    print("Duplicate article. Skip.")
else:
    print("Unique article. Publish.")
```
## 🛠 My Workflow in Practice
Here’s how I’ve integrated this into my project:
1. Every morning, I fetch trending news via RSS.
2. Each article is rewritten using AI.
3. The rewritten article is embedded using OpenAI and stored alongside the article content.
4. For every new article, I compare its embedding with all existing ones.
5. If the cosine similarity exceeds a chosen threshold (e.g. 0.85), the article is skipped.

> 💡 The threshold (e.g. 0.85) may need adjustment based on your domain. Try experimenting to find what works best.

## ⚠️ Real-World Considerations
While this system is powerful, no AI is perfect. Some notes to keep in mind:
- Long articles may need to be summarized before embedding.
- Two articles might differ structurally but still talk about the same facts.
- For efficiency, consider comparing only the introduction or summary of articles instead of full texts.

## ✅ Final Thoughts

Embeddings make your duplicate detection smarter by focusing on meaning, not just words. It’s a solid improvement over basic keyword matching — helping you reduce repetition, improve SEO, and provide more relevant content.

While not perfect, the approach is simple, efficient, and easy to plug into any automated workflow.