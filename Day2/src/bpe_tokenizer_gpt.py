from datasets import load_dataset
from tokenizers import (
    Tokenizer, models, pre_tokenizers, trainers, decoders, processors
)
from transformers import PreTrainedTokenizerFast


class BPEByteLevelTokenizer:
    """
    A class that encapsulates the training and usage of a Byte-Pair Encoding (BPE) byte-level tokenizer.
    
    This class includes functionality for:
    - Loading a dataset.
    - Training a tokenizer from the dataset.
    - Saving and loading a trained tokenizer.
    - Encoding and decoding text using the trained tokenizer.
    
    Methods:
    - __init__: Initializes the tokenizer with a Byte-Pair Encoding (BPE) model and byte-level pre-tokenizer.
    - prepare_training_corpus: Prepares the corpus from the dataset for training.
    - train_tokenizer: Trains the tokenizer using the training corpus.
    - save_tokenizer: Saves the trained tokenizer to a file.
    - load_tokenizer: Loads a previously saved tokenizer from a file.
    - encode_text: Encodes a given string of text using the trained tokenizer.
    - decode_ids: Decodes a list of token IDs back to the original string.
    - wrap_with_transformers: Wraps the tokenizer to be compatible with Hugging Face's transformers library.
    - run_pipeline: Executes the entire pipeline from preparation to tokenization in a clean function.
    """

    def __init__(self):
        """
        Initializes the BPEByteLevelTokenizer class.
        
        This method sets up the tokenizer with the BPE model and configures it to use byte-level pre-tokenization.
        The pre-tokenizer is set to handle tokenization at the byte level, splitting the input string into smaller chunks
        while maintaining the original offsets for proper alignment during encoding and decoding.
        """
        self.tokenizer = Tokenizer(models.BPE())  # Initialize the BPE tokenizer model
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # Set byte-level pre-tokenizer

    def prepare_training_corpus(self, dataset_name="wikitext", split="train", name="wikitext-2-raw-v1"):
        """
        Prepares a training corpus from a Hugging Face dataset.
        
        This method loads the dataset using the `load_dataset` function from the Hugging Face `datasets` library.
        It writes the dataset to a text file and generates chunks of data for use in training the tokenizer.
        
        Args:
            dataset_name (str): The name of the dataset to load. Defaults to 'wikitext'.
            split (str): The split of the dataset to use (e.g., 'train'). Defaults to 'train'.
            name (str): The specific name/version of the dataset. Defaults to 'wikitext-2-raw-v1'.
        """
        self.dataset = load_dataset(dataset_name, name=name, split=split)  # Load dataset

        # Save dataset to a file for later use
        with open("wikitext-2.txt", "w", encoding="utf-8") as f:
            for i in range(len(self.dataset)):
                f.write(self.dataset[i]["text"] + "\n")

        print("Training corpus saved to wikitext-2.txt")

    def get_training_corpus(self):
        """
        Yields chunks of the training corpus for the tokenizer to train on.
        
        This method creates chunks of 1000 sentences from the dataset to feed into the tokenizer during training.
        It yields these chunks as lists of strings to optimize memory usage and improve training performance.
        
        Returns:
            Generator: A generator that yields chunks of text from the dataset.
        """
        for i in range(0, len(self.dataset), 1000):
            yield self.dataset[i: i + 1000]["text"]

    def train_tokenizer(self, vocab_size=25000, special_tokens=["<|endoftext|>"]):
        """
        Trains the tokenizer on the provided corpus with a specified vocabulary size and special tokens.
        
        This method trains the tokenizer using the BPETrainer, which handles the learning of token merges and
        constructs a vocabulary based on the most frequent pairs of byte sequences in the corpus.
        
        Args:
            vocab_size (int): The size of the vocabulary to be built during training. Defaults to 25,000.
            special_tokens (list): A list of special tokens to include in the tokenizer's vocabulary. Defaults to ["<|endoftext|>"].
        """
        # Initialize BPETrainer for training
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

        # Train from the text file and from corpus chunks
        self.tokenizer.train(["wikitext-2.txt"], trainer=trainer)
        self.tokenizer.train_from_iterator(self.get_training_corpus(), trainer=trainer)

        print(f"Tokenizer trained with a vocabulary size of {vocab_size}.")

    def save_tokenizer(self, save_path="bpe_tokenizer_gpt"):
        """
        Saves the trained tokenizer to a specified path.
        
        This method writes the tokenizer model and vocabulary to disk, allowing it to be loaded later for encoding
        and decoding text without retraining.
        
        Args:
            save_path (str): The path where the tokenizer will be saved. Defaults to "bpe_tokenizer_gpt".
        """
        self.tokenizer.save(save_path)
        print(f"Tokenizer saved to {save_path}.")

    def load_tokenizer(self, load_path="bpe_tokenizer_gpt"):
        """
        Loads a previously saved tokenizer from disk.
        
        This method restores the tokenizer from a saved file, making it ready for use in encoding and decoding tasks.
        
        Args:
            load_path (str): The path to the saved tokenizer. Defaults to "bpe_tokenizer_gpt".
        """
        self.tokenizer = Tokenizer.from_file(load_path)
        print(f"Tokenizer loaded from {load_path}.")

    def encode_text(self, text):
        """
        Encodes a given text string into a list of token IDs.
        
        This method tokenizes the input string using the trained tokenizer and converts it into a list of integers
        representing the token IDs. These IDs can be used as input to machine learning models.
        
        Args:
            text (str): The input string to be encoded.
        
        Returns:
            list: A list of token IDs representing the encoded text.
        """
        encoding = self.tokenizer.encode(text)
        print(f"Text: '{text}'")
        print(f"Encoded tokens: {encoding.tokens}")
        return encoding.ids

    def decode_ids(self, token_ids):
        """
        Decodes a list of token IDs back into a human-readable string.
        
        This method takes a list of token IDs and converts them back into the original string using the tokenizer's
        decoding logic.
        
        Args:
            token_ids (list): A list of token IDs to be decoded.
        
        Returns:
            str: The decoded string.
        """
        decoded_text = self.tokenizer.decode(token_ids)
        print(f"Decoded text: '{decoded_text}'")
        return decoded_text

    def wrap_with_transformers(self, save_path="bpe_tokenizer_hf"):
        """
        Wraps the trained tokenizer to be compatible with Hugging Face's transformers library.
        
        This method converts the current tokenizer into a `PreTrainedTokenizerFast` and saves it in a format
        compatible with Hugging Face's model pipelines. The wrapped tokenizer includes special tokens for handling
        the beginning and end of sequences.
        
        Args:
            save_path (str): The path to save the wrapped tokenizer. Defaults to "bytebpe_tokenizer_pretrained".
        
        Returns:
            PreTrainedTokenizerFast: The wrapped tokenizer ready for use with transformers.
        """
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
        )
        wrapped_tokenizer.save_pretrained(save_path)
        print(f"Wrapped tokenizer saved to {save_path}.")
        return wrapped_tokenizer

    def run_pipeline(self):
        """
        Executes the entire tokenization pipeline, including:
        - Preparing the training corpus
        - Training the tokenizer
        - Saving the tokenizer
        - Loading the tokenizer
        - Encoding and decoding an example sentence
        - Wrapping and saving the tokenizer for Hugging Face's transformers
        
        This method provides a high-level, clean way to perform all the steps involved in the tokenizer training and usage process.
        """
        print("Starting the tokenization pipeline...")

        self.prepare_training_corpus()

        self.train_tokenizer()

        self.save_tokenizer()

        self.load_tokenizer()

        encoded_ids = self.encode_text("Let's test this tokenizer.")

        self.decode_ids(encoded_ids)

        self.wrap_with_transformers()

        print("Tokenization pipeline completed.")



if __name__ == "__main__":
    
    bpe_tokenizer = BPEByteLevelTokenizer()

    bpe_tokenizer.run_pipeline()
