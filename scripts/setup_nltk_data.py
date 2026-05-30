#!/usr/bin/env python3
"""Download the small NLTK resources used by the evaluation scripts."""
import nltk

for resource in ["punkt", "punkt_tab", "stopwords"]:
    print(f"Downloading {resource}...")
    nltk.download(resource)

print("NLTK setup complete.")
