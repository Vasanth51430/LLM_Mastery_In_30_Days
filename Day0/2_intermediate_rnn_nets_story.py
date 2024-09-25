import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Config:
    """
    Configuration class for text generation model.

    This class holds all the hyperparameters and configuration settings for the text generation model.
    It provides a centralized place to manage and adjust these parameters.

    Attributes:
        EMBEDDING_DIM (int): Dimension of the word embeddings. Higher dimensions can capture more 
                             nuanced relationships between words but require more computational resources.
        HIDDEN_DIM (int): Dimension of the hidden state in the RNN/LSTM/GRU. Larger hidden dimensions 
                          can model more complex patterns but increase the model's size and training time.
        NUM_LAYERS (int): Number of layers in the RNN/LSTM/GRU. More layers can capture hierarchical 
                          patterns in the data but are prone to vanishing gradients and overfitting.
        SEQ_LENGTH (int): Length of input sequences for training. Longer sequences provide more context 
                          but require more memory and can make training more difficult.
        BATCH_SIZE (int): Number of sequences in each batch during training. Larger batch sizes can lead 
                          to more stable gradients but require more memory.
        EPOCHS (int): Number of epochs to train the model. More epochs allow the model to see the data 
                      more times but can lead to overfitting if not properly regularized.
        LEARNING_RATE (float): Learning rate for the optimizer. Controls the step size during optimization. 
                               Too high can cause divergence, too low can result in slow convergence.
        UNK_TOKEN (str): Token to represent unknown words. Used to handle out-of-vocabulary words during 
                         inference.
    """

    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    SEQ_LENGTH = 10
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.001
    UNK_TOKEN = "<UNK>"

config = Config()

def tokenize_text(text):
    """
    Tokenize the input text into words.

    This function performs a simple space-based tokenization of the input text.
    It converts all words to lowercase to reduce vocabulary size and normalize the input.

    The tokenization process involves:
    1. Converting the entire text to lowercase.
    2. Splitting the text on whitespace (spaces, tabs, newlines).


    For more advanced tokenization, consider using libraries like NLTK or spaCy.

    Args:
        text (str): The input text to be tokenized. Can be a single sentence or multiple sentences.

    Returns:
        list: A list of lowercase tokens (words) from the input text.

    Example:
        >>> tokenize_text("Hello World! How are you?")
        ['hello', 'world!', 'how', 'are', 'you?']
    """
    return text.lower().split()

class StoryDataset(Dataset):
    """
    Dataset class for story text data.

    This class prepares the story data for training by creating sequences of tokens
    and their corresponding target tokens (next word prediction).

    The dataset is structured for next-word prediction tasks, where each input sequence
    is paired with a target sequence that is offset by one token.

    Attributes:
        data (list): List of token indices representing the story.
        seq_length (int): Length of input sequences.

    Args:
        data (list): List of token indices representing the story.
        seq_length (int): Length of input sequences.
    """

    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        """
        Get the total number of sequences in the dataset.

        The number of sequences is determined by the length of the data minus the sequence length.
        This ensures that each sequence has a corresponding target sequence of the same length.

        Returns:
            int: Number of possible sequences of length seq_length in the data.

        Example:
            If data has 100 tokens and seq_length is 10, __len__() will return 90.
        """
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        """
        Get a single input-target pair from the dataset.

        This method generates an input sequence and its corresponding target sequence.
        The target sequence is the input sequence shifted by one position, used for 
        next-word prediction training.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Input sequence of token indices.
                - torch.Tensor: Target sequence of token indices (shifted by one position).

        Example:
            If data is [1, 2, 3, 4, 5] and seq_length is 3:
            __getitem__(0) returns (tensor([1, 2, 3]), tensor([2, 3, 4]))
            __getitem__(1) returns (tensor([2, 3, 4]), tensor([3, 4, 5]))
        """
        return (torch.tensor(self.data[idx:idx + self.seq_length]), 
                torch.tensor(self.data[idx + 1:idx + self.seq_length + 1]))

class TextGenerationModel(nn.Module):
    """
    Text Generation Model using various RNN architectures.

    This class implements a text generation model that can use different types of
    recurrent neural networks (RNN, LSTM, GRU, BiLSTM) for sequence modeling.

    The model architecture consists of:
    1. An embedding layer to convert token indices to dense vectors.
    2. A recurrent layer (RNN/LSTM/GRU/BiLSTM) to process the sequence.
    3. A fully connected layer to project the RNN output to vocabulary size.

    Attributes:
        model_type (str): Type of RNN architecture to use.
        embedding (nn.Embedding): Word embedding layer.
        rnn (nn.Module): Recurrent neural network layer (RNN/LSTM/GRU/BiLSTM).
        fc (nn.Linear): Fully connected layer for output projection.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        hidden_dim (int): Dimension of the hidden state in the RNN.
        num_layers (int): Number of layers in the RNN.
        model_type (str): Type of RNN to use ('RNN', 'LSTM', 'GRU', or 'BiLSTM').
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, model_type="LSTM"):
        super(TextGenerationModel, self).__init__()
        self.model_type = model_type
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if model_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "BiLSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Unknown model type")
            
        if model_type == "BiLSTM":
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # Bidirectional LSTM has 2x hidden_dim
        else:
            self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        """
        Forward pass of the model.

        This method defines the forward pass of the text generation model:
        1. Convert input token indices to embeddings.
        2. Pass the embeddings through the RNN layer.
        3. Project the RNN output to vocabulary size using the fully connected layer.

        Args:
            x (torch.Tensor): Input tensor of token indices. Shape: (batch_size, sequence_length)
            hidden (torch.Tensor or tuple): Initial hidden state. 
                                            For LSTM: tuple of (hidden state, cell state)
                                            For others: tensor of hidden state

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor (logits for each token in the vocabulary).
                                Shape: (batch_size, sequence_length, vocab_size)
                - torch.Tensor or tuple: Final hidden state.
        """
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        """
        Initialize the hidden state for the RNN.

        This method creates the initial hidden state for the RNN layer. The shape and type
        of the hidden state depend on the RNN architecture being used.

        Args:
            batch_size (int): Batch size for initialization.

        Returns:
            torch.Tensor or tuple: Initial hidden state(s) for the RNN.
                - For LSTM and BiLSTM: tuple of (hidden state, cell state)
                - For RNN and GRU: tensor of hidden state

        Note:
            - For LSTM and BiLSTM, two tensors are returned: one for the hidden state and one for the cell state.
            - The shape of each tensor is (num_layers, batch_size, hidden_dim)
            - For bidirectional models, the number of layers is doubled.
        """
        if self.model_type == "LSTM" or self.model_type == "BiLSTM":
            return (torch.zeros(config.NUM_LAYERS, batch_size, config.HIDDEN_DIM),
                    torch.zeros(config.NUM_LAYERS, batch_size, config.HIDDEN_DIM))
        else:
            return torch.zeros(config.NUM_LAYERS, batch_size, config.HIDDEN_DIM)

def get_model(vocab_size, model_type="LSTM"):
    """
    Create and return an instance of the TextGenerationModel.

    This function serves as a factory method for creating TextGenerationModel instances.
    It uses the configuration parameters defined in the Config class to set up the model.

    Args:
        vocab_size (int): Size of the vocabulary. This determines the input and output 
                          dimensions of the model.
        model_type (str, optional): Type of RNN to use. Must be one of 'RNN', 'LSTM', 
                                    'GRU', or 'BiLSTM'. Defaults to "LSTM".

    Returns:
        TextGenerationModel: An instance of the text generation model configured with 
                             the specified parameters.
    """
    model = TextGenerationModel(vocab_size, config.EMBEDDING_DIM, config.HIDDEN_DIM, 
                                config.NUM_LAYERS, model_type=model_type)
    return model

def train_model(model, dataloader, epochs, lr):
    """
    Train the text generation model.

    This function trains the model using the provided dataloader for the specified number of epochs.
    It uses CrossEntropyLoss as the loss function and Adam as the optimizer.

    The training process includes:
    1. Iterating over the dataset for the specified number of epochs.
    2. For each batch:
       - Initialize (or reset) the hidden state.
       - Compute the model's output and loss.
       - Perform backpropagation and parameter updates.
    3. Printing the loss every 10 batches to monitor progress.

    Args:
        model (TextGenerationModel): The model to train.
        dataloader (DataLoader): DataLoader providing the training data.
        epochs (int): Number of epochs to train for.
        lr (float): Learning rate for the optimizer.

    Note:
        - This function assumes that the model and data are on the same device (CPU or GPU).
        - The hidden state is detached from the computation graph at each step to prevent 
          backpropagation through the entire sequence history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_size = inputs.size(0)  # Get actual batch size
            hidden = model.init_hidden(batch_size)  # Initialize hidden state with current batch size
            
            # Detach hidden states to prevent backprop through previous batches
            hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()
            
            optimizer.zero_grad()
            
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    print("Training complete.")

def generate_text(model, start_text, length, word2idx, idx2word):
    """
    Generate new text using the trained model.

    This function takes a starting text and generates 'length' number of new words
    using the trained model. It performs the following steps:
    1. Tokenize the start_text and convert tokens to indices.
    2. Initialize the model's hidden state.
    3. For each new word to generate:
       - Feed the current sequence through the model.
       - Select the most likely next word.
       - Append the new word to the generated text.
       - Update the input sequence for the next iteration.

    Args:
        model (TextGenerationModel): The trained text generation model.
        start_text (str): The initial text to start generation from.
        length (int): Number of words to generate.
        word2idx (dict): Mapping of words to their corresponding indices.
        idx2word (dict): Mapping of indices to their corresponding words.

    Returns:
        str: The generated text including the start_text.

    Note:
        - This function uses greedy decoding (always choosing the most probable next word).
          For more diverse outputs, consider implementing techniques like temperature 
          scaling or beam search.
        - Out-of-vocabulary words in the start_text are replaced with the UNK token.
        - The function assumes that the model is in evaluation mode (model.eval()).
    """
    model.eval()
    tokens = tokenize_text(start_text)
    
    # Handle OOV words by using the UNK token
    input_seq = torch.tensor(
        [word2idx.get(word, word2idx[config.UNK_TOKEN]) for word in tokens], 
        dtype=torch.long
    ).unsqueeze(0)
    
    hidden = model.init_hidden(1)
    
    generated_text = start_text
    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        next_word_idx = output.argmax(dim=2)[:, -1].item()
        next_word = idx2word[next_word_idx]
        
        generated_text += " " + next_word
        input_seq = torch.cat([input_seq, torch.tensor([[next_word_idx]])], dim=1)[:, -config.SEQ_LENGTH:]
    
    return generated_text

def prepare_dataset(story):
    """
    Prepare the dataset for training the text generation model.

    This function processes the input story text and creates the necessary data structures
    for training. It performs the following steps:
    1. Tokenize the story text.
    2. Create a vocabulary and mappings between words and indices.
    3. Convert the tokenized story to a list of indices.
    4. Create a StoryDataset instance and a DataLoader.

    Args:
        story (str): The input story text to be processed.

    Returns:
        tuple: A tuple containing:
            - dataloader (DataLoader): DataLoader for the prepared dataset.
            - vocab (list): List of unique words in the vocabulary.
            - word2idx (dict): Mapping of words to their corresponding indices.
            - idx2word (dict): Mapping of indices to their corresponding words.

    Note:
        - This function adds the UNK token to the vocabulary to handle out-of-vocabulary words.
        - The DataLoader is configured with the batch size and other parameters from the Config class.
    """
    tokenized_story = tokenize_text(story)
    vocab = sorted(set(tokenized_story))
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}

    # Add UNK token to vocabulary
    if config.UNK_TOKEN not in vocab:
        vocab.append(config.UNK_TOKEN)
        word2idx[config.UNK_TOKEN] = len(word2idx)
        idx2word[len(idx2word)] = config.UNK_TOKEN

    # Convert tokenized story to indices
    data = [word2idx[word] for word in tokenized_story]

    # Create dataset and dataloader
    dataset = StoryDataset(data, config.SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    return dataloader, vocab, word2idx, idx2word

def train_and_save_model(dataloader, vocab_size, model_type):
    """
    Train the text generation model and save it to a file.

    This function creates a model instance, trains it on the provided data,
    and saves the trained model to a file. It performs the following steps:
    1. Create a model instance using the specified model type.
    2. Train the model using the provided dataloader.
    3. Save the trained model to a file.

    Args:
        dataloader (DataLoader): DataLoader providing the training data.
        vocab_size (int): Size of the vocabulary.
        model_type (str): Type of RNN to use ('RNN', 'LSTM', 'GRU', or 'BiLSTM').

    Returns:
        TextGenerationModel: The trained model instance.
    """
    model = get_model(vocab_size, model_type)
    train_model(model, dataloader, config.EPOCHS, config.LEARNING_RATE)
    
    # Save the model
    torch.save(model.state_dict(), f"text_generation_{model_type}.pth")
    print(f"Model saved to text_generation_{model_type}.pth")
    
    return model

def load_model(vocab_size, model_type):
    """
    Load a previously trained model from a file.

    This function creates a new model instance and loads its parameters
    from a saved file. It's useful for loading a model for inference
    or further training.

    Args:
        vocab_size (int): Size of the vocabulary.
        model_type (str): Type of RNN used in the model ('RNN', 'LSTM', 'GRU', or 'BiLSTM').

    Returns:
        TextGenerationModel: The loaded model instance.
    """
    model = get_model(vocab_size, model_type)
    try:
        model.load_state_dict(torch.load(f"text_generation_{model_type}.pth"))
        print(f"Model loaded from text_generation_{model_type}.pth")
    except FileNotFoundError:
        print(f"Model file text_generation_{model_type}.pth not found. Please train the model first.")
        return None
    return model

def run_inference(model, start_text, length, word2idx, idx2word):
    """
    Run inference using the trained model to generate new text.

    This function generates new text using the provided model and starting text.
    It's a wrapper around the generate_text function that handles the setup and
    output formatting.

    Args:
        model (TextGenerationModel): The trained text generation model.
        start_text (str): The initial text to start generation from.
        length (int): Number of words to generate.
        word2idx (dict): Mapping of words to their corresponding indices.
        idx2word (dict): Mapping of indices to their corresponding words.

    Returns:
        None
    """
    generated_story = generate_text(model, start_text, length, word2idx, idx2word)
    print("Generated Story:\n", generated_story)

if __name__ == "__main__":

    # Dataset Collection (Simple Story Example)
    story = """
    Once upon a time, in a land far away, there was a peaceful village surrounded by mountains. 
    The villagers lived in harmony with nature. They grew crops, raised animals, and lived a simple but happy life.
    One day, a young girl named Lily discovered a mysterious cave hidden in the forest. She was curious and decided to explore.
    Inside the cave, she found glowing crystals and strange markings on the walls.
    As she ventured deeper, she realized she was not alone.
    """

    # Prepare the dataset
    dataloader, vocab, word2idx, idx2word = prepare_dataset(story)

    # Set the model type
    model_type = "LSTM"  # Change this to "RNN", "GRU", "BiLSTM" to experiment

    # Train and save the model
    model = train_and_save_model(dataloader, len(vocab), model_type)

    # Alternatively, load a previously trained model
    # model = load_model(len(vocab), model_type)

    # Run inference
    start_text = "once upon a time"
    run_inference(model, start_text, length=50, word2idx=word2idx, idx2word=idx2word)