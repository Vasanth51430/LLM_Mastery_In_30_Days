import unicodedata

def get_stats(ids, counts=None):
    """
    Compute the frequency of consecutive integer pairs in a list.

    Args:
        ids (list of int): A list of integers, representing token IDs, typically in the range 0-255.
        counts (dict, optional): An existing dictionary where consecutive pairs are already counted.
                                 If None, an empty dictionary is used.

    Returns:
        dict: A dictionary where the keys are 2-tuples of consecutive integers from `ids`, and
              the values are the count (frequency) of how often each pair appears in the list.

    Details:
        - This function generates all consecutive pairs from the list `ids`, meaning it will take 
          each element in `ids` and pair it with the next element.
        - The dictionary `counts` is updated with these pairs as keys and their frequency as values.
        - For example, if `ids = [1, 2, 3, 1, 2]`, the pairs generated would be:
          (1, 2), (2, 3), (3, 1), (1, 2), and the output will be:
          {(1, 2): 2, (2, 3): 1, (3, 1): 1}.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    Replace occurrences of a specific consecutive pair in a list of integers with a new token.

    Args:
        ids (list of int): A list of integers where consecutive pairs might be merged into a new token.
        pair (tuple of int): A 2-tuple representing the consecutive pair of integers to be merged.
        idx (int): The integer token that will replace occurrences of the `pair` in `ids`.

    Returns:
        list of int: A new list of integers where every occurrence of `pair` in `ids` is replaced by `idx`.

    Details:
        - The function scans the list `ids` from left to right.
        - When the function finds a consecutive occurrence of `pair` in `ids`, it replaces both elements
          in the pair with the new integer `idx`.
        - The function skips over both elements in the pair when it performs a replacement.
        - For example, if `ids = [1, 2, 3, 1, 2]`, and `pair = (1, 2)`, and `idx = 4`, 
          the output will be `[4, 3, 4]`, where both (1, 2) pairs are replaced by 4.
    """
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    """
    Replaces Unicode control characters in a string with their escaped Unicode representations.

    Args:
        s (str): The input string, which may contain control characters.

    Returns:
        str: The input string, with all control characters replaced by their Unicode escape sequences.

    Details:
        - Control characters are non-printable characters, like newline (`\\n`), tab (`\\t`), etc.
        - The function uses `unicodedata.category()` to check the Unicode category of each character in the string.
        - Control characters (those categorized with a "C" prefix in Unicode) are replaced with their Unicode 
          code point, formatted as `\\uXXXX`, where `XXXX` is the hexadecimal representation of the character.
        - Non-control characters are left unchanged.
        - For example, if `s = 'Hello\\nWorld!'`, the result would be `'Hello\\u000aWorld!'`, replacing the newline with its Unicode escape.
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    """
    Decodes a byte token to a human-readable string, handling control characters.

    Args:
        t (bytes): A token in byte form, representing either part of a UTF-8 encoded string 
                   or a special token like raw bytes (0-255).

    Returns:
        str: A human-readable string where control characters are escaped and non-printable byte sequences 
             are safely decoded into a UTF-8 string.

    Details:
        - The function attempts to decode the byte sequence into a UTF-8 string. If the bytes
          contain invalid UTF-8 sequences, they will be replaced with the Unicode replacement character `�`.
        - The function then calls `replace_control_characters()` to handle control characters.
        - For example, `t = b'Hello\\x00World!'` would be decoded into `'Hello\\u0000World!'`, 
          where `\\x00` is a null byte (non-printable) and is escaped.
    """
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s