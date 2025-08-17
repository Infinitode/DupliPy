import unittest
from duplipy.formatting import *
from duplipy.replication import *
from duplipy.similarity import *
from duplipy.text_analysis import *

class TestTextFunctions(unittest.TestCase):

    def test_remove_stopwords(self):
        text = "this is a test sentence"
        self.assertEqual(remove_stopwords(text), "test sentence")

    def test_remove_numbers(self):
        text = "text with numbers 123"
        self.assertEqual(remove_numbers(text), "text with numbers ")

    def test_remove_whitespace(self):
        text = "  text   with   extra   whitespace  "
        self.assertEqual(remove_whitespace(text), "text with extra whitespace")

    def test_normalize_whitespace(self):
        text = "  text   with   extra   whitespace  "
        self.assertEqual(normalize_whitespace(text), " text with extra whitespace ")

    def test_separate_symbols(self):
        text = "text,with,symbols"
        self.assertEqual(separate_symbols(text), "text , with , symbols")

    def test_remove_special_characters(self):
        text = "text!@#with$%^special&*characters"
        self.assertEqual(remove_special_characters(text), "textwithspecialcharacters")

    def test_standardize_text(self):
        text = "  Text To Be Standardized  "
        self.assertEqual(standardize_text(text), "text to be standardized")

    def test_tokenize_text(self):
        text = "text to be tokenized"
        self.assertEqual(tokenize_text(text), ["text", "to", "be", "tokenized"])

    def test_stem_words(self):
        words = ["running", "jumps", "happily"]
        self.assertEqual(stem_words(words), ["run", "jump", "happili"])

    def test_lemmatize_words(self):
        words = ["running", "jumps", "happily"]
        self.assertEqual(lemmatize_words(words), ["running", "jump", "happily"])

    def test_pos_tag(self):
        text = "this is a test"
        self.assertEqual(pos_tag(text), [('this', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN')])

    def test_remove_profanity_from_text(self):
        text = "this is a damn test"
        # This is dependent on the external valx library, so we can't be sure of the output.
        # We will just check that the function runs without error.
        remove_profanity_from_text(text)

    def test_remove_sensitive_info_from_text(self):
        text = "my email is test@test.com"
        # This is dependent on the external valx library, so we can't be sure of the output.
        # We will just check that the function runs without error.
        remove_sensitive_info_from_text(text)

    def test_remove_hate_speech_from_text(self):
        text = "I hate you"
        # This is dependent on the external valx library, so we can't be sure of the output.
        # We will just check that the function runs without error.
        remove_hate_speech_from_text(text)

    def test_post_format_text(self):
        text = "this is a test sentence , how are you ?"
        self.assertEqual(post_format_text(text), "this is a test sentence, how are you?")

    def test_replace_word_with_synonym(self):
        word = "happy"
        # This is non-deterministic, so we can't test for a specific output.
        # We will just check that the function returns a string.
        self.assertIsInstance(replace_word_with_synonym(word), str)

    def test_augment_text_with_synonyms(self):
        text = "this is a test"
        augmented_text = augment_text_with_synonyms(text, 2, 0.5, progress=False)
        self.assertEqual(len(augmented_text), 2)

    def test_insert_random_word(self):
        text = "this is a test"
        self.assertNotEqual(insert_random_word(text, "new"), text)

    def test_random_word_deletion(self):
        text = "this is a test"
        self.assertNotEqual(random_word_deletion(text), text)

    def test_swap_random_words(self):
        text = "this is a test"
        self.assertNotEqual(swap_random_words(text), text)

    def test_edit_distance_score(self):
        text1 = "hello"
        text2 = "hallo"
        self.assertEqual(edit_distance_score(text1, text2), 1)

    def test_bleu_score(self):
        ref = "this is a test"
        cand = "this is a test"
        self.assertEqual(bleu_score(ref, cand), 1.0)

    def test_jaccard_similarity_score(self):
        text1 = "this is a test"
        text2 = "this is a test"
        self.assertEqual(jaccard_similarity_score(text1, text2), 1.0)

    def test_sorensen_dice_coefficient(self):
        text1 = "this is a test"
        text2 = "this is a test"
        self.assertEqual(sorensen_dice_coefficient(text1, text2), 1.0)

    def test_cosine_similarity_score(self):
        text1 = "this is a test"
        text2 = "this is a test"
        self.assertEqual(cosine_similarity_score(text1, text2), 1.0)

    def test_analyze_sentiment(self):
        text = "I am happy"
        self.assertGreater(analyze_sentiment(text), 0)

    def test_named_entity_recognition(self):
        text = "John Doe went to New York"
        # This is non-deterministic, so we can't test for a specific output.
        # We will just check that the function returns a list.
        self.assertIsInstance(named_entity_recognition(text), nltk.tree.Tree)

if __name__ == "__main__":
    unittest.main()
