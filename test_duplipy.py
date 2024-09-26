from duplipy import *

text = "The quick    brown fox jumps over the lazy dog.545 g w 4       "
print("Removed stopwords: ", remove_stopwords(text))
print("Removed numbers: ", remove_numbers(text))
print("Removed whitespace: ", remove_whitespace(text))
print("Normalize whitespace: ", normalize_whitespace(text))
print("Seperate symbols: ", separate_symbols(text))
print("Remove special characters: ", remove_special_characters(text))
print("Standardize text: ", standardize_text(text))
print("Tokenize text: ", tokenize_text(text))
print("POS tag: ", pos_tag(text))
print("Insert random word (call): ", insert_random_word(text, "call"))
print("Delete random word: ", delete_random_word(text))
print("Insert synonym (touch): ", insert_synonym(text, "feel"))
print("Paraphrased: ", paraphrase(text))

print("Edit distance score between Hi and Hello: ", edit_distance_score("hi", "hello"))
print("BLEU score calculation: ", bleu_score("Hello, how are you?", "Hi, how are you doing?"))

print("Analyze sentiment: ", analyze_sentiment(text))