"""
Text replication for NLP.

Available functions:
- `replace_word_with_synonym(word)`: Replace the given word with a synonym.
- `augment_text_with_synonyms(text, augmentation_factor, probability, progress=True)`: Augment the input text by replacing words with synonyms.
- `load_text_file(filepath)`: Load the contents of a text file.
- `augment_file_with_synonyms(file_path, augmentation_factor, probability, progress=True)`: Augment a text file by replacing words with synonyms.
- `insert_random_word(text, word)`: Insert a random word into the input text.
- `delete_random_word(text)`: Delete a random word from the input text.
- `random_word_deletion(text, num_deletions=1)`: Deletes a user-specified number of random words from the text.
- `swap_random_words(text)`: Swaps two random words in the text.
- `insert_synonym(text, word)`: Insert a synonym of the given word into the input text.
- `paraphrase(text)`: Paraphrase the input text.
- `flip_horizontal(image)`: Flip the input image horizontally.
- `flip_vertical(image)`: Flip the input image vertically.
- `rotate(image, angle)`: Rotate the input image by a specified angle.
- `random_rotation(image, max_angle)`: Randomly rotate the input image by an angle within the specified range.
- `resize(image, size)`: Resize the input image to the specified size.
- `crop(image, box)`: Crop the input image to the specified rectangular region.
- `random_crop(image, size)`: Randomly crop a region from the input image.
- `shuffle_words(text)`: Randomly shuffle the order of words in each sentence.
- `random_flip(image, horizontal, vertical)`: Randomly flip the input image horizontally and/or vertically.
- `random_color_jitter(image, brightness, contrast, saturation, hue)`: Randomly adjust the brightness, contrast, saturation, and hue of the input image.
- `noise_overlay(image, noise_factor, noise_type, grain_factor)`: Overlay noise on the input image.
"""

import random
import nltk
import numpy as np
from nltk.corpus import wordnet
from PIL import Image, ImageEnhance
from tqdm import tqdm
from typing import Any
from datetime import datetime, timedelta
import csv
import os

def replace_word_with_synonym(word: str) -> str:
    """
    Replace the given word with a synonym.

    Synonyms are alternative words with similar meanings, and replacing words
    with synonyms can be used for text augmentation or variation.
    
    Parameters:
    - `word` (str): The input word to replace with a synonym.

    Returns:
    - `str`: The synonym for the word.
    """
    try:
        nltk.download("wordnet", quiet=True)
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        
        if synonyms:
            synonym = random.choice(synonyms)
            return str(synonym)
        
        return word
    except Exception as e:
        print(f"An error occurred during word replacement: {str(e)}")
        return word

def augment_text_with_synonyms(text: str, augmentation_factor: int, probability: float, progress: bool = True) -> list[str]:
    """
    Augment the input text by replacing words with synonyms.

    Parameters:
    - `text` (str): The input text to be augmented.
    - `augmentation_factor` (int): The number of times to augment the text.
    - `probability` (float): The probability of replacing a random word with a synonym.
    - `progress` (bool): Whether or not to return current progress during augmentation.

    Returns:
    - `list[str]`: A list of augmented text strings.
    """
    augmented_text = []
    try:
        if probability is None:
            raise ValueError("Probability value cannot be of NoneType. Choose a float from 0 to 1")

        tokens = text.split()

        with tqdm(total=augmentation_factor * len(tokens), desc="Augmenting Text", disable=not progress) as pbar:
            for _ in range(augmentation_factor):
                augmented_tokens = []

                for token in tokens:
                    if random.random() < probability:
                        replaced_token = replace_word_with_synonym(token)
                        augmented_tokens.append(replaced_token)
                    else:
                        augmented_tokens.append(token)
                    pbar.update(1)

                augmented_text.append(' '.join(augmented_tokens))

    except Exception as e:
        print(f"An error occurred during text augmentation: {str(e)}")
        return []

    return augmented_text

def load_text_file(file_path: str) -> str:
    """
    Load the contents of a text file.

    Parameters:
    - `file_path` (str): The path to the target input data.

    Returns:
    - `str`: The read text from the file.
    """
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"An error occurred during text file loading: {str(e)}")
        return ""

def augment_file_with_synonyms(file_path: str, augmentation_factor: int, probability: float, progress: bool = True) -> list[str]:
    """
    Augment a text file by replacing words with synonyms.

    Parameters:
    - `file_path` (str): The path to the target input data.
    - `augmentation_factor` (int): The number of times to augment the data.
    - `probability` (float): The probability of replacing a random word with its synonym.
    - `progress` (bool): Whether or not to print the current progress during augmentation.

    Returns:
    - `list[str]`: A list of augmented text strings.
    """
    try:
        text = load_text_file(file_path)
        augmented_text = augment_text_with_synonyms(text, augmentation_factor, probability, progress)
        return augmented_text
    except Exception as e:
        print(f"An error occurred during text file augmentation: {str(e)}")
        return []


def insert_random_word(text: str, word: str) -> str:
    """
    Insert a random word into the input text.

    This function randomly inserts a specified word into the input text, creating
    variations for text augmentation or diversification.

    Parameters:
    - `text` (str): The input text for word insertion.
    - `word` (str): The word to be inserted into the text.

    Returns:
    - `str`: The text with the randomly inserted word.
    """
    try:
        nltk.download("punkt", quiet=True)
        words = nltk.word_tokenize(text)
        words.insert(random.randint(0, len(words)), word)
        modified_text = " ".join(words)
        return modified_text
    except Exception as e:
        print(f"An error occurred during word insertion: {str(e)}")
        return text


def random_word_deletion(text: str, num_deletions: int = 1) -> str:
    """
    Delete a random word from the input text.

    This function randomly deletes a word from the input text, creating variations
    for text augmentation or diversity.
    
    Parameters:
    - `text` (str): The input text for word deletion.
    - `num_deletions` (int): The number of words to delete.

    Returns:
    - `str`: The text with a randomly deleted word.
    """
    try:
        nltk.download("punkt", quiet=True)
        words = nltk.word_tokenize(text)
        for _ in range(num_deletions):
            if len(words) > 1:
                words.pop(random.randint(0, len(words) - 1))
        modified_text = " ".join(words)
        return modified_text
    except Exception as e:
        print(f"An error occurred during word deletion: {str(e)}")
        return text

def delete_random_word(text: str) -> str:
    """
    Delete a random word from the input text.

    This function randomly deletes a word from the input text, creating variations
    for text augmentation or diversity.

    Parameters:
    - `text` (str): The input text for word deletion.

    Returns:
    - `str`: The text with a randomly deleted word.
    """
    return random_word_deletion(text, num_deletions=1)

def swap_random_words(text: str) -> str:
    """
    Swaps two random words in the text.

    This function randomly swaps two words in the input text, creating variations
    for text augmentation or diversity.

    Parameters:
    - `text` (str): The input text for word swapping.

    Returns:
    - `str`: The text with two words swapped.
    """
    try:
        nltk.download("punkt", quiet=True)
        words = nltk.word_tokenize(text)
        if len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        modified_text = " ".join(words)
        return modified_text
    except Exception as e:
        print(f"An error occurred during word swapping: {str(e)}")
        return text

def insert_synonym(text: str, word: str) -> str:
    """
    Insert a synonym of the given word into the input text.

    This function replaces the specified word in the input text with a synonym,
    introducing variations for text augmentation or diversity.
    
    Parameters:
    - `text` (str): The input text for synonym insertion.
    - `word` (str): The word for which a synonym will be inserted.

    Returns:
    - `str`: The text with a synonym of the word inserted.
    """
    try:
        synonym = replace_word_with_synonym(word)
        modified_text = text.replace(word, synonym)
        return modified_text
    except Exception as e:
        print(f"An error occurred during synonym insertion: {str(e)}")
        return text


def paraphrase(text: str) -> str:
    """
    Paraphrase the input text.

    This function leverages part-of-speech tagging to identify verbs (VB), nouns (NN),
    and adjectives (JJ) in the input text, replacing them with synonyms for paraphrasing.
    
    Parameters:
    - `text` (str): The input text to be paraphrased.

    Returns:
    - `str`: The paraphrased text.
    """
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        paraphrased_tokens = [replace_word_with_synonym(token) if tag.startswith(("VB", "NN", "JJ")) else token for token, tag in tagged_tokens]
        paraphrased_text = " ".join(paraphrased_tokens)
        return paraphrased_text
    except Exception as e:
        print(f"An error occurred during paraphrasing: {str(e)}")
        return text

def flip_horizontal(image: Image.Image) -> Image.Image:
    """
    Flip the input image horizontally.

    Parameters:
    - `image` (PIL.Image.Image): The input image to be flipped.

    Returns:
    - `PIL.Image.Image`: The horizontally flipped image.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(image: Image.Image) -> Image.Image:
    """
    Flip the input image vertically.

    Parameters:
    - `image` (PIL.Image.Image): The input image to be flipped.

    Returns:
    - `PIL.Image.Image`: The vertically flipped image.
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def rotate(image: Image.Image, angle: float) -> Image.Image:
    """
    Rotate the input image by a specified angle.

    Parameters:
    - `image` (PIL.Image.Image): The input image to be rotated.
    - `angle` (float): The angle of rotation in degrees.

    Returns:
    - `PIL.Image.Image`: The rotated image.
    """
    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    return crop(rotated_image, (0, 0, *image.size))

def random_rotation(image: Image.Image, max_angle: float = 30) -> Image.Image:
    """
    Randomly rotate the input image by an angle within the specified range.

    Parameters:
    - `image` (PIL.Image.Image): The input image to be randomly rotated.
    - `max_angle` (float, optional): The maximum absolute angle of rotation in degrees. Default is 30.

    Returns:
    - `PIL.Image.Image`: The randomly rotated image.
    """
    angle = random.uniform(-max_angle, max_angle)
    return rotate(image, angle)

def resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Resize the input image to the specified size.

    Parameters:
    - `image` (PIL.Image.Image): The input image to be resized.
    - `size` (tuple[int, int]): The new size in the format (width, height).

    Returns:
    - `PIL.Image.Image`: The resized image.
    """
    return image.resize(size, Image.BICUBIC)

def crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """
    Crop the input image to the specified rectangular region.

    Parameters:
    - `image` (PIL.Image.Image): The input image to be cropped.
    - `box` (tuple[int, int, int, int]): A tuple (left, upper, right, lower) specifying the region to crop.

    Returns:
    - `PIL.Image.Image`: The cropped image.
    """
    return image.crop(box)

def random_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Randomly crop a region from the input image.

    Parameters:
    - `image` (PIL.Image.Image): The input image from which to extract the random crop.
    - `size` (tuple[int, int]): The size of the output crop in the format (width, height).

    Returns:
    - `PIL.Image.Image`: The randomly cropped image region.
    """
    width, height = image.size
    left = random.randint(0, width - size[0])
    upper = random.randint(0, height - size[1])
    right = left + size[0]
    lower = upper + size[1]
    return crop(image, (left, upper, right, lower))

# DupliPy 0.2.0

def shuffle_words(text: list[str]) -> list[str]:
    """
    Randomly shuffle the order of words in each sentence.

    This function takes a list of sentences and randomly shuffles the order of words
    in each sentence, creating variations for text augmentation or diversity.
    
    Parameters:
    - `text` (list[str]): List of sentences where each sentence's words needs to be shuffled.

    Returns:
    - `list[str]`: List of sentences with randomly shuffled words.
    """
    # Shuffle the order of words in each sentence
    shuffled_text = []
    with tqdm(total=len(text), desc="Shuffling Words") as pbar:
        for sentence in text:
            words = sentence.split()
            shuffled_words = random.sample(words, len(words))
            shuffled_sentence = ' '.join(shuffled_words)
            shuffled_text.append(shuffled_sentence)
            pbar.update(1)
    return shuffled_text

def random_flip(image: Image.Image, horizontal: bool = True, vertical: bool = True) -> Image.Image:
    """
    Randomly flip the input image horizontally and/or vertically.

    Parameters:
    - `image` (PIL.Image.Image): The input image to be flipped.
    - `horizontal` (bool): Whether to flip the image horizontally.
    - `vertical` (bool): Whether to flip the image vertically.

    Returns:
    - `PIL.Image.Image`: The randomly flipped image.
    """
    if horizontal and vertical:
        flip = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180])
    elif horizontal:
        flip = Image.FLIP_LEFT_RIGHT
    elif vertical:
        flip = Image.FLIP_TOP_BOTTOM
    else:
        return image

    return image.transpose(flip)

def random_color_jitter(image: Image.Image, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1) -> Image.Image:
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input image.

    Parameters:
    - `image` (PIL.Image.Image): The input image to be color-jittered.
    - `brightness` (float): The maximum factor to adjust brightness.
    - `contrast` (float): The maximum factor to adjust contrast.
    - `saturation` (float): The maximum factor to adjust saturation.
    - `hue` (float): The maximum factor to adjust hue.

    Returns:
    - `PIL.Image.Image`: The color-jittered image.
    """
    image = ImageEnhance.Brightness(image).enhance(1 + random.uniform(-brightness, brightness))
    image = ImageEnhance.Contrast(image).enhance(1 + random.uniform(-contrast, contrast))
    image = ImageEnhance.Color(image).enhance(1 + random.uniform(-saturation, saturation))

    h, s, v = image.convert("HSV").split()
    hue_factor = int(255 * random.uniform(-hue, hue))
    h = h.point(lambda i: (i + hue_factor) % 256)
    image = Image.merge("HSV", (h, s, v)).convert("RGB")

    return image

def noise_overlay(image: Image.Image, noise_factor: float = 0.1, noise_type: str = "gaussian", grain_factor: float = 0.0) -> Image.Image:
    """
    Overlay noise on the input image.

    Parameters:
        - `image` (PIL.Image.Image): The input image to overlay noise on.
        - `noise_factor` (float): The factor to control the intensity of the noise (0.0 to 1.0).
        - `noise_type` (str): The type of noise to overlay ("gaussian", "salt_and_pepper"). Defaults to "gaussian".
        - `grain_factor` (float): The factor to control the graininess of the noise (0.0 to 1.0). Defaults to 0.0.

    Returns:
        - `PIL.Image.Image`: The image with overlaid noise.
    """
    noise = Image.new("RGB", image.size)

    if noise_type == "gaussian":
        # Generate random Gaussian noise with mean 128 and standard deviation proportional to noise_factor
        for x in range(noise.width):
            for y in range(noise.height):
                noise_value = int(128 + random.gauss(0, noise_factor * 255))
                noise_value = max(0, min(255, noise_value))
                noise.putpixel((x, y), (noise_value, noise_value, noise_value))
    elif noise_type == "salt_and_pepper":
        # Generate salt and pepper noise with probability proportional to noise_factor
        for x in range(noise.width):
            for y in range(noise.height):
                if random.random() < noise_factor:
                    noise_value = 0 if random.random() < 0.5 else 255
                    noise.putpixel((x, y), (noise_value, noise_value, noise_value))
                else:
                    # Keep original or neutral? noise_overlay seems to blend later.
                    # In the original code it was just filling the 'noise' image.
                    pass
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")

    # Add grain effect by scaling random noise and blending with original image
    grain_noise = Image.new("RGB", image.size)
    for x in range(grain_noise.width):
        for y in range(grain_noise.height):
            noise_val = int(random.uniform(-grain_factor * 255, grain_factor * 255))
            # Just some variation
            grain_noise.putpixel((x, y), (noise_val, noise_val, noise_val))

    blended_noise = Image.blend(noise, grain_noise, grain_factor)

    return Image.blend(image, blended_noise, noise_factor)

# DupliPy 0.2.6

def add_noise(data: list[float] | np.ndarray, noise_factor: float = 0.05) -> list[float] | np.ndarray:
    """
    Add random Gaussian noise to numerical data.

    Parameters:
    - `data` (list[float] | np.ndarray): The input numerical data.
    - `noise_factor` (float): The factor to control the intensity of the noise.

    Returns:
    - `list[float] | np.ndarray`: The data with added noise.
    """
    try:
        arr = np.array(data, dtype=float)
        std = np.std(arr)
        if std == 0:
            std = 1.0
        noise = np.random.normal(0, noise_factor * std, arr.shape)
        augmented = arr + noise
        return augmented.tolist() if isinstance(data, list) else augmented
    except Exception as e:
        print(f"An error occurred during noise addition: {str(e)}")
        return data

def scale_data(data: list[float] | np.ndarray, scaling_factor: float = 1.1) -> list[float] | np.ndarray:
    """
    Scale numerical data by a factor.

    Parameters:
    - `data` (list[float] | np.ndarray): The input numerical data.
    - `scaling_factor` (float): The factor to scale the data by.

    Returns:
    - `list[float] | np.ndarray`: The scaled data.
    """
    try:
        arr = np.array(data, dtype=float)
        augmented = arr * scaling_factor
        return augmented.tolist() if isinstance(data, list) else augmented
    except Exception as e:
        print(f"An error occurred during data scaling: {str(e)}")
        return data

def shift_data(data: list[float] | np.ndarray, shift_amount: float = 1.0) -> list[float] | np.ndarray:
    """
    Shift numerical data by a constant amount.

    Parameters:
    - `data` (list[float] | np.ndarray): The input numerical data.
    - `shift_amount` (float): The amount to shift the data by.

    Returns:
    - `list[float] | np.ndarray`: The shifted data.
    """
    try:
        arr = np.array(data, dtype=float)
        augmented = arr + shift_amount
        return augmented.tolist() if isinstance(data, list) else augmented
    except Exception as e:
        print(f"An error occurred during data shifting: {str(e)}")
        return data

def _parse_date(date_str: str) -> datetime | None:
    """Helper to parse common date formats."""
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def augment_time_series(data: list[Any], augmentation_factor: int = 1) -> list[Any]:
    """
    Augment time series data by generating new timestamps within the range.

    Parameters:
    - `data` (list[Any]): The input time series data (strings or datetime objects).
    - `augmentation_factor` (int): How many new points to generate per existing point.

    Returns:
    - `list[Any]`: The augmented time series data.
    """
    try:
        dates = []
        is_string = False
        for item in data:
            if isinstance(item, str):
                is_string = True
                dt = _parse_date(item)
                if dt:
                    dates.append(dt)
            elif isinstance(item, datetime):
                dates.append(item)

        if len(dates) < 2:
            return data

        dates.sort()
        min_date = dates[0]
        max_date = dates[-1]
        total_seconds = (max_date - min_date).total_seconds()

        new_dates = []
        for _ in range(len(data) * augmentation_factor):
            random_seconds = random.uniform(0, total_seconds)
            new_dt = min_date + timedelta(seconds=random_seconds)
            if is_string:
                new_dates.append(new_dt.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                new_dates.append(new_dt)

        return data + new_dates
    except Exception as e:
        print(f"An error occurred during time series augmentation: {str(e)}")
        return data

def balance_dataset(rows: list[dict[str, Any]], target_column: str) -> list[dict[str, Any]]:
    """
    Balance a dataset based on a target categorical column by oversampling minority classes.

    Parameters:
    - `rows` (list[dict[str, Any]]): The input data as a list of dictionaries.
    - `target_column` (str): The column name to balance by.

    Returns:
    - `list[dict[str, Any]]`: The balanced dataset.
    """
    try:
        if not rows:
            return rows

        counts = {}
        for row in rows:
            val = row.get(target_column)
            counts[val] = counts.get(val, 0) + 1

        max_count = max(counts.values())
        balanced_rows = list(rows)

        for val, count in counts.items():
            if count < max_count:
                needed = max_count - count
                pool = [r for r in rows if r.get(target_column) == val]
                for _ in range(needed):
                    balanced_rows.append(random.choice(pool).copy())

        return balanced_rows
    except Exception as e:
        print(f"An error occurred during dataset balancing: {str(e)}")
        return rows

def augment_csv_data(input_path: str, output_path: str, augmentation_factor: int = 1, balance_column: str | None = None, fill_missing: bool = True) -> bool:
    """
    Augment and expand CSV data. Performs imputation, optional balancing, and expansion.

    Parameters:
    - `input_path` (str): Path to the input CSV file.
    - `output_path` (str): Path where the augmented CSV will be saved.
    - `augmentation_factor` (int): How many new entries to generate relative to the original.
    - `balance_column` (str | None): Optional column name to balance the dataset by.
    - `fill_missing` (bool): Whether to fill missing values with column modes or means.

    Returns:
    - `bool`: True if successful, False otherwise.
    """
    try:
        rows = []
        with open(input_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

        if not rows:
            return False

        if fill_missing:
            for col in fieldnames:
                vals = [r[col] for r in rows if r[col]]
                if not vals:
                    continue
                try:
                    # Try numeric mean
                    num_vals = [float(v) for v in vals]
                    fill_val = str(sum(num_vals) / len(num_vals))
                except ValueError:
                    # Categorical mode
                    fill_val = max(set(vals), key=vals.count)

                for r in rows:
                    if not r[col]:
                        r[col] = fill_val

        if balance_column:
            rows = balance_dataset(rows, balance_column)

        augmented_rows = list(rows)
        for _ in range(augmentation_factor):
            for row in rows:
                new_row = row.copy()
                for col in fieldnames:
                    val = new_row[col]
                    try:
                        # Numeric jitter
                        num_val = float(val)
                        new_row[col] = str(num_val * random.uniform(0.95, 1.05))
                    except ValueError:
                        # Check if date
                        dt = _parse_date(val)
                        if dt:
                            # Random shift +/- 1 day
                            new_dt = dt + timedelta(seconds=random.randint(-86400, 86400))
                            new_row[col] = new_dt.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            # Categorical: swap with another existing value in that column with low probability
                            if random.random() < 0.1:
                                new_row[col] = random.choice([r[col] for r in rows])
                augmented_rows.append(new_row)

        with open(output_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(augmented_rows)

        return True
    except Exception as e:
        print(f"An error occurred during CSV augmentation: {str(e)}")
        return False
