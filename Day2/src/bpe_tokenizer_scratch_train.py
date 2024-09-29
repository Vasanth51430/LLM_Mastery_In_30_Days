from bpe_tokenizer_scratch import BPETokenizer
import os

corpus_file = '../res/corpus.txt' 
tokenizer_save_prefix = 'bpe_tokenizer' 


def train_tokenizer(corpus_file: str, vocab_size: int, save_prefix: str):
    """
    Train the BasicTokenizer on the text corpus and save the tokenizer.

    Args:
        corpus_file (str): Path to the text corpus file.
        vocab_size (int): The target vocabulary size for the tokenizer.
        save_prefix (str): The file prefix for saving the tokenizer model and vocab files.

    Returns:
        None: Saves the tokenizer model (.model) and vocab (.vocab) to disk.
    """
    with open(corpus_file, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = BPETokenizer()

    tokenizer.train(text, vocab_size=vocab_size, verbose=True)

    tokenizer.save(save_prefix)
    print(f"Tokenizer saved to {save_prefix}.model and {save_prefix}.vocab")

# 3. Create the function to load the tokenizer and tokenize a new sentence
def load_and_tokenize(model_file: str, new_sentence: str):
    """
    Load the BasicTokenizer from a saved model file and tokenize a new sentence.

    Args:
        model_file (str): Path to the saved tokenizer model file (.model).
        new_sentence (str): The new sentence to tokenize.

    Returns:
        list: A list of token IDs representing the new sentence.
    """
    tokenizer = BPETokenizer()

    tokenizer.load(model_file)
    print(f"Tokenizer loaded from {model_file}")

    token_ids = tokenizer.encode(new_sentence)
    print(f"Token IDs for the sentence '{new_sentence}': {token_ids}")
    
    return token_ids

if __name__ == "__main__":
    if not os.path.exists(f"{tokenizer_save_prefix}.model"):
        print("Training the tokenizer on the corpus...")
        train_tokenizer(corpus_file, vocab_size=300, save_prefix=tokenizer_save_prefix)
    else:
        print("Tokenizer already trained and saved.")

    new_sentence = "Hello world!"
    token_ids = load_and_tokenize(f"{tokenizer_save_prefix}.model", new_sentence)
    print(f"Tokenized sentence: {token_ids}")
