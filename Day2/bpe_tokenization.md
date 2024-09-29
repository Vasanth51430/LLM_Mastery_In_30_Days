# BPE & BYTE LEVEL BPE TOKENIZATION

## BYTE PAIR ENCODING

### **1. What is Byte-Pair Encoding (BPE)?**
- Originally, BPE was used in data compression to replace frequent pairs of bytes with a single byte.
- In **natural language processing (NLP)**, BPE has been adapted to merge frequent character pairs in text. Instead of bytes, it merges characters or character sequences.

### **2. Pre-Tokenization**
- **Pre-tokenization** splits the text into words based on spaces. After splitting, a special symbol `_` is added at the end of each word to mark word boundaries.
  
**Example Text**:  
“low low low low lower lower newest newest widest widest”

Pre-tokenized text becomes:  
`low_`, `lower_`, `newest_`, `widest_`

- Count the frequency of each word. Here, we get:
  - `low_`: 5 times
  - `lower_`: 2 times
  - `newest_`: 6 times
  - `widest_`: 3 times

This list of words and frequencies is the starting point for the BPE algorithm.

### **3. Vocabulary Construction**
This step involves iteratively creating a vocabulary by merging frequent pairs of characters (or symbols). Here's how it works:

#### **Step 1: Create Base Vocabulary**
- First, extract all unique characters from the text. The initial vocabulary includes these individual characters:
  - `l, o, w, e, r, n, s, t, i, d, _`

#### **Step 2: Represent Words Using the Base Vocabulary**
- Break each word into its individual characters (tokens):
  - `low_`: `(l, o, w, _)`
  - `lower_`: `(l, o, w, e, r, _)`
  - `newest_`: `(n, e, w, e, s, t, _)`
  - `widest_`: `(w, i, d, e, s, t, _)`

#### **Step 3: Merge the Most Frequent Pairs**
Now, BPE merges the most frequent pair of characters. This is done iteratively:

- **Merge 1**: The most frequent pair is `(e, s)`, occurring 9 times (6 in `newest_` and 3 in `widest_`).
  - After merging `e` and `s`, the vocabulary is updated to include the new symbol `es`.
  - Words are updated:
    - `newest_` → `(n, e, w, es, t, _)`
    - `widest_` → `(w, i, d, es, t, _)`

- **Merge 2**: Next, merge `es` and `t`, which appears 9 times.
  - A new symbol `est` is created, updating the vocabulary.
  - Words are updated:
    - `newest_` → `(n, e, w, est, _)`
    - `widest_` → `(w, i, d, est, _)`

- **Merge 3**: Merge `est` and `_`, forming the new symbol `est_`.
  - Words are updated:
    - `newest_` → `(n, e, w, est_)`
    - `widest_` → `(w, i, d, est_)`

- **Merge 4**: Merge the pair `l, o` (from `low_` and `lower_`), forming the new symbol `lo`.
  - Words are updated:
    - `low_` → `(lo, w, _)`
    - `lower_` → `(lo, w, e, r, _)`

- **Merge 5**: Merge the pair `lo` and `w`, forming the new symbol `low`.
  - Words are updated:
    - `low_` → `(low, _)`
    - `lower_` → `(low, e, r, _)`

#### **Step 4: Stop When Vocabulary Size is Reached**
- Keep merging until the vocabulary reaches the desired size. After these 5 merges, we have the following vocabulary:
  - `l, o, w, e, r, n, s, t, i, d, _, es, est, est_, lo, low`

The merge rules are:
- `(e, s)` → `es`
- `(es, t)` → `est`
- `(est, _)` → `est_`
- `(l, o)` → `lo`
- `(lo, w)` → `low`

---

### **4. Tokenizing New Text**
Once the vocabulary is built, we can tokenize new text using the learned merge rules.

#### **Step 1: Pre-Tokenization**
New text is first split into words and each word is followed by the `_` symbol.  
**Example Text**:  
“newest binded lowers”  
Pre-tokenized as:  
`newest_`, `binded_`, `lowers_`

#### **Step 2: Apply Merge Rules**
Break the words into characters and apply the learned merge rules:
- `newest_`: Breaks into `(n, e, w, e, s, t, _)`
- Apply `(e, s)` → `es`:  
  `(n, e, w, es, t, _)`
- Apply `(es, t)` → `est`:  
  `(n, e, w, est, _)`
- Apply `(est, _)` → `est_`:  
  Final token: `(n, e, w, est_)`

- `binded_`: Breaks into `(b, i, n, d, e, d, _)`
- Since there is no merge rule for this sequence, it remains unchanged.

- `lowers_`: Breaks into `(l, o, w, e, r, s, _)`
- Apply `(l, o)` → `lo`:  
  `(lo, w, e, r, s, _)`
- Apply `(lo, w)` → `low`:  
  Final token: `(low, e, r, s, _)`

#### **Step 3: Handle Unknown Tokens**
If any token is not found in the vocabulary, it is replaced with an `[UNK]` token.  
For example, since `binded_` contains no known merge rules, part of it would be marked as unknown.

---

### **5. Final Tokenization Result**
For the text “newest binded lowers”, the tokenization result is:  
- `newest` → `[n, e, w, est_]`
- `binded` → `[UNK]`
- `lowers` → `[low, e, r, s, _]`

### **Conclusion**
Byte-Pair Encoding is a simple and efficient method for creating subword tokens by merging frequently occurring character pairs. It helps NLP models handle rare and unknown words effectively by breaking them down into subword units.

## BYTE LEVEL BPE

Let's break it down step by step to understand how **byte-level BPE** works, with a focus on **Tiktoken** as used in GPT models.


### **1. What is Byte-Level BPE?**
- Instead of working on characters or words, **byte-level BPE** operates at the **byte** level, meaning it works directly with the raw bytes (the numerical representations of characters in a computer). This allows it to handle all Unicode characters, including emojis, accented characters, and other special symbols.
- Byte-level BPE ensures that any character can be encoded, regardless of its encoding (like UTF-8). This is especially useful for languages with many unique characters (such as Chinese or Arabic) and for handling edge cases like unseen symbols or control characters.

### **2. Why Use Byte-Level BPE for GPT?**
- GPT models need to process a vast variety of languages, punctuation marks, emojis, and even unseen or rare symbols.
- By working at the byte level, the tokenizer can efficiently handle any kind of text input without needing to worry about how different characters are encoded or represented.
- This also allows the tokenization process to be **lossless**—meaning no information is lost even when working with rare or special characters.

### **3. How Byte-Level BPE Works in Tiktoken**

Here’s how the process happens step by step:

#### **Step 1: Convert Text to Byte Sequence**
Instead of starting with characters, the input text is first converted into a sequence of bytes. This means each character is represented by its corresponding byte in the **UTF-8** encoding.

**Example**:  
Let’s take the word **“apple”**. Each character is converted into its byte representation based on UTF-8:
- `a` → 97
- `p` → 112
- `p` → 112
- `l` → 108
- `e` → 101

So, the word "apple" would be represented as a byte sequence like:
```
[97, 112, 112, 108, 101]
```

This byte-level representation ensures that any character (even non-ASCII ones) is captured without losing information.

#### **Step 2: Apply BPE on the Byte Level**
Once the input is converted to bytes, **BPE merges frequent byte pairs**, just like the character-level BPE we discussed earlier.

##### **Initial Vocabulary**
- The initial vocabulary for byte-level BPE contains **all possible 256 byte values** (from 0 to 255). These represent every possible character or symbol in the byte range.
- Additionally, it contains single-character tokens (i.e., tokens for individual bytes).

##### **Merging Byte Pairs**
- The BPE algorithm then iteratively merges **the most frequent adjacent byte pairs** in the text. Each time it merges a pair, it creates a new token that represents that pair.

**Example**:  
Let’s say you have a sequence like `112, 112` (for the letters `pp`). The algorithm might decide to merge `112, 112` into a single new token, which we'll call `T_pp`.

If this pair appears frequently, it becomes part of the vocabulary, and the word **“apple”** could now be represented as:
```
[97, T_pp, 108, 101] → [a, pp, l, e]
```

After several merges, more complex sequences are added to the vocabulary (e.g., entire words or subword units). This allows the tokenizer to efficiently represent frequent patterns of text.

#### **Step 3: Tokenization**
After building the vocabulary through byte-level BPE, the tokenizer can now break down any text into subword tokens based on this learned vocabulary.

**Example**:  
If the final vocabulary contains tokens for `"apple"`, `"app"`, and `"le"`, the word **"apple"** might be tokenized as:
```
["app", "le"]
```

And the byte-level sequence will be encoded as:
```
[<token for "app">, <token for "le">]
```

This results in compact tokenization while still capturing the full meaning of the word.

---

### **4. Handling Special Characters and Emojis**
One of the key benefits of byte-level BPE is its ability to handle **special characters, emojis, and multi-byte Unicode characters**. Since BPE operates at the byte level, it can split even complex characters into byte sequences and merge them based on frequency.

**Example**:  
Consider the emoji 😊. The UTF-8 byte representation for this emoji is:
```
[240, 159, 152, 138]
```
With byte-level BPE, this emoji is first split into these bytes, and as the vocabulary grows, these byte sequences may be merged into a single token representing the emoji.

This flexibility allows GPT models to understand and tokenize even the most complex characters without any additional processing or special cases.

---

### **5. Final Tokenization Result and Benefits**

In the case of byte-level BPE in Tiktoken (as used in GPT), the final tokenization works as follows:
- **All possible byte values** are represented in the vocabulary (allowing for complete coverage of text).
- The tokenizer breaks down text into byte sequences and merges frequent byte pairs to generate tokens.
- The final tokenization is efficient and compact, especially for frequent patterns of text, and can handle special symbols, emojis, and multilingual input easily.

**Benefits of Byte-Level BPE**:
- **Universal coverage**: It can encode any input text, regardless of language, script, or symbols.
- **Efficient representation**: Frequently occurring subwords or entire words are merged, reducing the number of tokens required to represent common patterns.
- **Lossless encoding**: Since it works at the byte level, no information is lost during tokenization, even for rare or unseen characters.

---

### **6. Example: How GPT Models Use Byte-Level BPE**

Let's say GPT is processing a sentence like **"GPT-3 is amazing 🚀!"**. Here's how the tokenization happens:
1. **Convert the sentence to bytes**:
   - `G` → `71`, `P` → `80`, `T` → `84`, `-` → `45`, `3` → `51`, and so on.
   - The emoji 🚀 becomes its UTF-8 byte sequence: `[240, 159, 154, 128]`.

2. **Apply byte-level BPE**:
   - Byte pairs like `71, 80` (for "GP") or `240, 159` (for the first part of the rocket emoji) might be merged based on frequency.

3. **Final tokens**:
   - After merging, common words or subwords like `"GPT"` or `"amazing"` might be represented by single tokens, while the rocket emoji might be represented as another single token.

---

### **Conclusion**
Byte-level BPE as used in Tiktoken for GPT tokenization is a highly flexible and efficient method for processing any type of text, no matter the language or characters. By starting with bytes and merging frequent pairs, it builds an efficient vocabulary that can represent both common and rare text patterns, including special characters and emojis, in a way that preserves all the input information.