from transformers import MarianMTModel, MarianTokenizer
import sys
from nltk.tokenize import sent_tokenize

# Load the model and tokenizer.
mt_model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
mt_model = MarianMTModel.from_pretrained(mt_model_name)

print(f"Ready.", file=sys.stderr)
for text in sys.stdin:
    # Break the text into sentences.
    for sentence in sent_tokenize(text):

        inputs = tokenizer(sentence, return_tensors="pt")
        translated = mt_model.generate(**inputs)

        # Decode and print the translated sentences.
        decoded = tokenizer.decode(translated, 
                                   skip_special_tokens=True)
        for sentence in decoded:
            print(sentence)

print("Finished.", file=sys.stderr)