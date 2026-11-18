import unittest
from PIL import Image
from duplipy.similarity import sorensen_dice_coefficient, mean_squared_error, psnr
from duplipy._nltk_utils import safe_word_tokenize, safe_sent_tokenize


class TestNewFixes(unittest.TestCase):
    def test_sorensen_dice_empty_strings(self):
        self.assertEqual(sorensen_dice_coefficient("", ""), 1.0)
        self.assertEqual(sorensen_dice_coefficient("hello", ""), 0.0)

    def test_mean_squared_error_rgb_and_mismatch(self):
        img1 = Image.new("RGB", (10, 10), color=(0, 0, 0))
        img2 = Image.new("RGB", (10, 10), color=(255, 255, 255))
        self.assertAlmostEqual(mean_squared_error(img1, img2), 255.0 ** 2)
        self.assertAlmostEqual(psnr(img1, img1), 100.0)

        img_diff = Image.new("RGB", (5, 5), color=(0, 0, 0))
        with self.assertRaises(ValueError):
            mean_squared_error(img1, img_diff)

    def test_safe_tokenizers(self):
        self.assertEqual(safe_word_tokenize("hello world"), ["hello", "world"])
        self.assertEqual(
            safe_sent_tokenize("Hello world. How are you?"),
            ["Hello world.", "How are you?"],
        )


if __name__ == "__main__":
    unittest.main()
