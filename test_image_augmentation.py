import unittest
import os
from PIL import Image
from duplipy.replication import flip_horizontal, flip_vertical, rotate, random_rotation, resize, crop, random_crop

class TestImageAugmentation(unittest.TestCase):

    def setUp(self):
        self.test_image_path = "./test_image.jpg"
        self.test_image = Image.new("RGB", (500, 500), color="red")
        self.test_image.save(self.test_image_path)

    def tearDown(self):
        os.remove(self.test_image_path)

    def test_flip_horizontal(self):
        flipped_image = flip_horizontal(self.test_image)
        self.assertTrue(isinstance(flipped_image, Image.Image))
        self.assertEqual(flipped_image.size, self.test_image.size)

    def test_flip_vertical(self):
        flipped_image = flip_vertical(self.test_image)
        self.assertTrue(isinstance(flipped_image, Image.Image))
        self.assertEqual(flipped_image.size, self.test_image.size)

    def test_rotate(self):
        rotated_image = rotate(self.test_image, 45)
        self.assertTrue(isinstance(rotated_image, Image.Image))
        self.assertEqual(rotated_image.size, self.test_image.size)

    def test_random_rotation(self):
        random_rotated_image = random_rotation(self.test_image, max_angle=30)
        self.assertTrue(isinstance(random_rotated_image, Image.Image))
        self.assertEqual(random_rotated_image.size, self.test_image.size)

    def test_resize(self):
        resized_image = resize(self.test_image, (50, 50))
        self.assertTrue(isinstance(resized_image, Image.Image))
        self.assertEqual(resized_image.size, (50, 50))

    def test_crop(self):
        cropped_image = crop(self.test_image, (25, 25, 75, 75))
        self.assertTrue(isinstance(cropped_image, Image.Image))
        self.assertEqual(cropped_image.size, (50, 50))

    def test_random_crop(self):
        random_cropped_image = random_crop(self.test_image, (50, 50))
        self.assertTrue(isinstance(random_cropped_image, Image.Image))
        self.assertEqual(random_cropped_image.size, (50, 50))


if __name__ == "__main__":
    unittest.main()