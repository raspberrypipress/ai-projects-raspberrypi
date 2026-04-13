from transformers import MarianMTModel, MarianTokenizer
import time
import sys

if __name__ == "__main__":

    mt_model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    mt_model = MarianMTModel.from_pretrained(mt_model_name)

    print(f"CTRL+C to stop...", file=sys.stderr)
    try:
        for line in sys.stdin:
            inputs = tokenizer(line.strip(),
                               return_tensors="pt",
                               padding=True,
                               truncation=True)
            translated = mt_model.generate(**inputs)
            r = tokenizer.decode(translated, 
                                 skip_special_tokens=True)
            print(r[0])
    except KeyboardInterrupt:
        print("Finished!", file=sys.stderr)
