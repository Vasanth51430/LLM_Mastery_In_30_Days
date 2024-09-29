from base_tokenizer import Tokenizer
from bpe_tokenizer_scratch_utils import *

class BPETokenizer(Tokenizer):
    """
    Basic byte-level Byte Pair Encoding (BPE) tokenizer that follows the core
    principles of the GPT tokenizer with a simplified approach.

    This tokenizer is trained on raw bytes of input text and uses byte-level
    BPE to generate tokens. It iteratively merges the most frequent consecutive
    byte pairs until the vocabulary reaches the desired size.

    Attributes:
        merges (dict): Dictionary mapping byte-pair tuples to new token IDs.
        vocab (dict): A dictionary where the keys are token IDs and the values
                      are corresponding byte sequences.
    """

    def __init__(self):
        """
        Initialize a BasicTokenizer instance.

        Inherits the base `Tokenizer` class and sets up the tokenizer's default attributes.
        - Initializes an empty merge rule set and a base vocabulary of byte-level tokens (256 tokens).
        """
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Trains the byte-level BPE tokenizer on the provided text, constructing a vocabulary of the specified size.

        Args:
            text (str): The input text to train on.
                        This text is converted to its byte representation for BPE training.
            vocab_size (int): Desired size of the vocabulary.
                              Must be greater than or equal to 256 (the number of byte-level tokens).
            verbose (bool): If True, prints each merge step for debugging and visibility into the training process.
                            Default is False.

        Returns:
            None: This method modifies the instance's `merges` and `vocab` attributes directly.

        Raises:
            AssertionError: If `vocab_size` is less than 256 (the number of byte-level tokens).

        Detailed Steps:
        - Convert the input string `text` into a list of byte-level integers (each byte is a number between 0-255).
        - Iteratively merge the most common consecutive byte pairs in the `ids` list until the `vocab_size` is reached.
        - Each time a pair is merged, a new token is created and assigned the next available ID starting from 256.
        - The training is complete once the desired number of merges (vocab_size - 256) is performed.

        Example:
            >>> tokenizer = BasicTokenizer()
            >>> tokenizer.train("hello world", vocab_size=300)
            >>> print(tokenizer.vocab)  # A dictionary with token IDs and their corresponding byte sequences.
        """
        assert vocab_size >= 256, "Vocab size must be at least 256 to cover all byte-level tokens."
        num_merges = vocab_size - 256

        # Convert input text to its byte representation
        text_bytes = text.encode("utf-8")  # raw bytes of the input text
        ids = list(text_bytes)  # convert bytes to a list of integers (byte-level tokens)

        # Create the initial vocabulary (byte-level tokens)
        merges = {}  # dictionary to store merges: (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # start with byte-level tokens (0-255)

        # Perform iterative merging to create new tokens
        for i in range(num_merges):
            # Count the frequency of consecutive byte pairs
            stats = get_stats(ids)
            # Find the most frequent pair to merge
            pair = max(stats, key=stats.get)
            # Create a new token ID for this merged pair
            idx = 256 + i
            # Replace all occurrences of the pair in the ids list with the new token
            ids = merge(ids, pair, idx)
            # Record the merge in the merges dictionary
            merges[pair] = idx
            # Update the vocabulary with the new token
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # Optionally, print the merging process if verbose mode is enabled
            if verbose:
                print(f"Merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # Save the merges and vocabulary as class attributes for future encoding/decoding
        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> list:
        """
        Encodes a given text string into a list of token IDs using the trained BPE model.

        Args:
            text (str): The input text to encode.

        Returns:
            list: A list of token IDs representing the input text.

        Detailed Steps:
        - The input text is first converted to byte-level integers (using UTF-8 encoding).
        - The tokenizer then repeatedly merges the most frequent byte pairs based on the learned `merges`.
        - The resulting list of token IDs is returned after no more valid merges can be applied.

        Example:
            >>> tokenizer = BasicTokenizer()
            >>> tokenizer.train("hello world", vocab_size=300)
            >>> token_ids = tokenizer.encode("hello")
            >>> print(token_ids)
            [104, 101, 108, 108, 111]  # Byte-level tokens or merged tokens depending on the vocabulary
        """
        # Convert input text to byte-level tokens
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # Continuously merge pairs according to the learned merge rules
        while len(ids) >= 2:
            # Get frequency stats for consecutive pairs
            stats = get_stats(ids)
            # Find the best pair to merge (has the lowest merge index)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # If no valid merges exist, break the loop
            if pair not in self.merges:
                break

            # Otherwise, apply the merge and update the ids
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def decode(self, ids: list) -> str:
        """
        Decodes a list of token IDs back into a human-readable string.

        Args:
            ids (list): A list of token IDs representing a previously encoded text.

        Returns:
            str: The decoded string, converted back from byte-level tokens.

        Detailed Steps:
        - The list of token IDs is first converted back to a byte sequence by looking up the corresponding byte values
          in the vocabulary.
        - The byte sequence is then decoded back into a UTF-8 string.
        - Any invalid UTF-8 sequences are replaced with the replacement character (�).

        Example:
            >>> tokenizer = BasicTokenizer()
            >>> tokenizer.train("hello world", vocab_size=300)
            >>> token_ids = tokenizer.encode("hello")
            >>> text = tokenizer.decode(token_ids)
            >>> print(text)
            'hello'
        """
        # Convert token IDs back to byte sequences
        text_bytes = b"".join(self.vocab[idx] for idx in ids)

        # Decode byte sequence to a string, replacing invalid sequences
        text = text_bytes.decode("utf-8", errors="replace")
        return text
