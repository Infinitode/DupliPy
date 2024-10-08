# DupliPy 0.2.3
![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![Code Size](https://img.shields.io/github/languages/code-size/infinitode/duplipy)
![Downloads](https://pepy.tech/badge/duplipy)
![License Compliance](https://img.shields.io/badge/license-compliance-brightgreen.svg)
![PyPI Version](https://img.shields.io/pypi/v/duplipy)

An open source Python library for text formatting, augmentation, and similarity calculation tasks in NLP, the package now also includes additional methods for image augmentation.

## Changes to DupliPy 0.2.3

Duplipy now utilizes another one of our Python packages, called ValX, which provides quick methods we can use to clean and format our text data before training in preprocessing steps.

## Installation

You can install DupliPy using pip:

```bash
pip install duplipy
```

## Supported Python Versions

DupliPy supports the following Python versions:

- Python 3.6
- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12 or later

Please ensure that you have one of these Python versions installed before using DupliPy. DupliPy may not work as expected on lower versions of Python than the supported.

## Features

- Text Formatting: Remove special characters, standardize text formatting.
- Text Replication: Generate replicated instances of text for data augmentation.
- Sentiment Analysis: Find impressions within sentences.
- Similarity Calculation: Calculate text similarity using various metrics.
- BLEU Score Calculation: Calculate how well your text-based NLP model performs.
- Image Augmentation Tasks.
- Profanity removal, hate speech removal, offensive speech removal, and sensitive information removal.

*For full reference documentation view [DupliPy's official documentation](https://infinitode-docs.gitbook.io/documentation/package-documentation/duplipy-package-documentation).*

## Usage

### Text Formatting

```python
from duplipy.formatting import remove_special_characters, standardize_text

text = "Hello! This is some text."

# Remove special characters
formatted_text = remove_special_characters(text)
print(formatted_text)  # Output: Hello This is some text

# Standardize text formatting
standardized_text = standardize_text(text)
print(standardized_text)  # Output: hello! this is some text
```

### Text Replication

```python
from duplipy.replication import replace_word_with_synonym, augment_text_with_synonyms

text = "Hello! This is some text."

# Replace words with synonyms
augmented_text = augment_text_with_synonyms(text, augmentation_factor=3, probability=0.5)
print(augmented_text)

# Output:
# ['Hello! This is some text.', 'Hi! This is some text.', 'Hello! This is certain text.']

# Load text from a file and augment it
file_path = "path/to/file.txt"
augmented_file_text = augment_file_with_synonyms(file_path, augmentation_factor=3, probability=0.5)
print(augmented_file_text)
```

### Sentiment Analysis

```python
from duplipy.text_analysis import analyze_sentiment

text = "I love this product! It's amazing!"

# Analyze sentiment
sentiment = analyze_sentiment(text)
print(sentiment)  # Output: Positive
```

### Similarity Calculation

```python
from duplipy.similarity import edit_distance_score

text1 = "Hello! How are you?"
text2 = "Hi! How are you doing?"

# Calculate edit distance
edit_distance = edit_distance_score(text1, text2)
print(edit_distance)  # Output: 4
```

### BLEU Score Calculation

```python
from duplipy.similarity import bleu_score

text1 = "Hello! How are you?"
text2 = "Hi! How are you doing?"

# Calculate cosine similarity
bleu_value = bleu_score(text1, text2)
print(bleu_value)  # Output: 0.434
```

### Image Augmentation

```python
from PIL import Image
from duplipy.replication import flip_horizontal, flip_vertical, rotate, random_rotation, resize, crop, random_crop

# Load an image for testing
image_path = "path/to/image.jpg"
image = Image.open(image_path)

# Flip the image horizontally
flipped_horizontal_image = flip_horizontal(image)

# Flip the image vertically
flipped_vertical_image = flip_vertical(image)

# Rotate the image by a specific angle (e.g., 45 degrees)
rotated_image = rotate(image, 45)

# Apply a random rotation to the image within a specified range of angles (e.g., -30 to 30 degrees)
randomly_rotated_image = random_rotation(image, max_angle=30)

# Resize the image to a specific target size (e.g., 224x224 pixels)
resized_image = resize(image, target_size=(224, 224))

# Crop a random region from the image (e.g., 150x150 pixels)
randomly_cropped_image = random_crop(image, crop_size=(150, 150))

# Save the augmented images (optional, if you want to view the results)
flipped_horizontal_image.save("path/to/flipped_horizontal.jpg")
flipped_vertical_image.save("path/to/flipped_vertical.jpg")
rotated_image.save("path/to/rotated.jpg")
randomly_rotated_image.save("path/to/randomly_rotated.jpg")
resized_image.save("path/to/resized.jpg")
randomly_cropped_image.save("path/to/randomly_cropped.jpg")
```

### Hate speech and Offensive speech removal using AI

```python
from duplipy.formatting import remove_hate_speech_from_text

text = "I hate all of you bad word! Can't you just bad word leave me alone! Hi, I'm Katy."

print(remove_hate_speech_from_text(text))

### Output
# "Hi, I'm Katy."
```

## Contributing

Contributions are welcome! If you encounter any issues, have suggestions, or want to contribute to DupliPy, please open an issue or submit a pull request on [GitHub](https://github.com/infinitode/duplipy).

## License

DupliPy is released under the terms of the **MIT License (Modified)**. Please see the [LICENSE](https://github.com/infinitode/duplipy/blob/main/LICENSE) file for the full text.

**Modified License Clause**



The modified license clause grants users the permission to make derivative works based on the DupliPy software. However, it requires any substantial changes to the software to be clearly distinguished from the original work and distributed under a different name.

By enforcing this distinction, it aims to prevent direct publishing of the source code without changes while allowing users to create derivative works that incorporate the code but are not exactly the same.

Please read the full license terms in the [LICENSE](https://github.com/infinitode/duplipy/blob/main/LICENSE) file for complete details.
