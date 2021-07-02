from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("cahya/t5-base-indonesian-summarization-cased")
model = T5ForConditionalGeneration.from_pretrained("cahya/t5-base-indonesian-summarization-cased")

def get_sumary(article):
  input_ids = tokenizer.encode(article, return_tensors='pt')
  summary_ids = model.generate(input_ids,
            min_length=20,
            max_length=80,
            num_beams=10,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True,
            do_sample = True,
            temperature = 0.8,
            top_k = 50,
            top_p = 0.95)
  summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  print(summary_text)

  inp = input("Masukan artikel : ")
get_sumary(inp)