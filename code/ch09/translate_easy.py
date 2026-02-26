#!/usr/bin/env python3
from easynmt import EasyNMT
import time

model = EasyNMT('opus-mt')

text = """
Artificial intelligence is an umbrella term that is — in itself — rather vague. It covers everything from very specific technologies, such as machine learning, deep learning, and natural language processing through to the use of AI technologies in hardware and software projects. But it’s also a science fiction term that pits human intelligence against perceived computer intelligence. 
It’s for this reason that many AI experts use the term artificial intelligence carefully and tend to home in on detailed technologies with distinct meanings such as machine learning or natural language processing.
"""
start = time.perf_counter()
translation = model.translate(text, target_lang='es')
elapsed = time.perf_counter() - start

print(translation)
print(f'Translation took: {elapsed:.6f} seconds')
