from transformers import MarianMTModel, MarianTokenizer
import sys
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":

    # Load the model and tokenizer.
    mt_model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    mt_model = MarianMTModel.from_pretrained(mt_model_name)

    print(f"CTRL+C to stop...", file=sys.stderr)
    try:
        for text in sys.stdin:
            # Break the input text into sentences.
            sentences = sent_tokenize(text)

            # Translate the sentences as a batch.
            inputs = tokenizer(sentences,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               )
            translated = mt_model.generate(**inputs)

            # Decode and print the translated sentences.
            decoded = tokenizer.decode(translated, 
                                       skip_special_tokens=True
                                      )
            for sentence in decoded:
                print(sentence)

    except KeyboardInterrupt:
        print("Finished!", file=sys.stderr)
