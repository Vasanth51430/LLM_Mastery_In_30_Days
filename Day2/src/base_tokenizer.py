import unicodedata

class Tokenizer:
    """Base class for tokenizers that implements vocabulary management and saving/loading functionality."""

    def __init__(self):
        """
        Initializes the tokenizer with a basic byte-level vocabulary (0-255).
        
        Details:
        - The `merges` dictionary tracks which byte pairs (tuples of integers) have been merged into new tokens.
        - The `special_tokens` dictionary allows certain reserved tokens (e.g., for marking the end of a sentence)
          to be used with their own IDs.
        - The initial vocabulary is built by calling `_build_vocab()`, which assigns each byte (0-255) its
          corresponding byte value.
        """
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        """
        Trains the tokenizer on text to build a vocabulary of a specified size.

        Args:
            text (str): The input text to train on, which will be tokenized at the byte level.
            vocab_size (int): The target vocabulary size, which includes both byte-level tokens 
                              and any additional merged tokens.
            verbose (bool, optional): If True, prints details about the merging process during training.

        Details:
        - Not yet implemented in the base class. Meant to be overridden in subclasses.
        """
        raise NotImplementedError

    def encode(self, text):
        """
        Encodes a given string into a sequence of token IDs (integers).
        
        Args:
            text (str): The input string to encode.
            
        Returns:
            list of int: The encoded sequence of token IDs.
            
        Details:
        - Not yet implemented in the base class. Meant to be overridden in subclasses.
        """
        raise NotImplementedError

    def decode(self, ids):
        """
        Decodes a sequence of token IDs (integers) into a string.
        
        Args:
            ids (list of int): The sequence of token IDs to decode.
            
        Returns:
            str: The decoded string.
            
        Details:
        - Not yet implemented in the base class. Meant to be overridden in subclasses.
        """
        raise NotImplementedError

    def _build_vocab(self):
        """
        Builds the initial vocabulary based on the byte-level tokens (0-255) and any merged tokens.

        Returns:
            dict: A dictionary mapping token IDs (integers) to their byte representations.
        
        Details:
        - For each byte (0-255), it creates a single-byte token.
        - For each merged token (created during training), it concatenates the byte sequences of the
          two merged tokens.
        - Special tokens, if defined, are also added to the vocabulary with their respective byte encodings.
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves the tokenizer model and vocabulary to files.

        Args:
            file_prefix (str): The file path prefix. Two files will be created: 
                               one for the model (with a `.model` extension) and 
                               one for the vocabulary (with a `.vocab` extension).

        Details:
        - The model file stores essential information for reloading the tokenizer, such as version,
          merge patterns, and special tokens.
        - The vocabulary file is a human-readable representation of the tokens and merges. 
          It cannot be used to restore the tokenizer because decoding is lossy (some byte sequences
          may not be valid UTF-8 strings).
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            # Write special tokens to the model file
            for token, idx in self.special_tokens.items():
                f.write(f"{idx} {token}\n")
            # Write merges to the model file
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} -> {idx}\n")

        vocab_file = file_prefix + ".vocab"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token_bytes in self.vocab.items():
                token_str = token_bytes.decode('utf-8', errors='replace')
                print(token_str)
                print(idx)
                f.write(f"{idx}: {token_str}\n")

    def load(self, file_prefix):
        """
        Loads the tokenizer model and vocabulary from files.

        Args:
            file_prefix (str): The file path prefix for the files to load.
                               It expects two files: one with a `.model` extension for the model
                               and another with a `.vocab` extension for the vocabulary.

        Details:
        - The model file restores the tokenizer configuration such as the merging rules and special tokens.
        - The vocabulary file is used only for verification purposes, but the core tokenizer's
          functionality is restored using the model file alone.
        """
        model_file = file_prefix
        with open(model_file, 'r') as f:
            lines = f.readlines()
            assert lines[0].strip() == "minbpe v1", "Unknown model format"
            self.pattern = lines[1].strip()
            num_special_tokens = int(lines[2].strip())
            
            # Read special tokens
            self.special_tokens = {}
            current_line = 3
            for _ in range(num_special_tokens):
                idx, token = lines[current_line].strip().split(" ", 1)
                self.special_tokens[token] = int(idx)
                current_line += 1

            # Read merges
            self.merges = {}
            for line in lines[current_line:]:
                p0, p1, _, idx = line.split()
                self.merges[(int(p0), int(p1))] = int(idx)
        
        # Rebuild the vocabulary from the loaded model data
        self.vocab = self._build_vocab()