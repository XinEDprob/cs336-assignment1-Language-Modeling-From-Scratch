import os
import regex as re
import logging
from typing import BinaryIO
from collections import Counter, defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import json
import base64
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


RAW_TEXT_FOLDER_PATH = "/Users/xshi849/Documents/playground/cs336-assignment1-Language-Modeling-From-Scratch/data"
RAW_TEXT_NAME = "TinyStoriesV2-GPT4-valid.txt"
RAW_TEXT_PATH = RAW_TEXT_FOLDER_PATH + "/" + RAW_TEXT_NAME

TRAINED_DATA_FOLDER = "/Users/xshi849/Documents/playground/cs336-assignment1-Language-Modeling-From-Scratch/trained"
TRAINED_BPE_PICKLE = TRAINED_DATA_FOLDER + "/" + RAW_TEXT_NAME.split(".")[0] + "_trained_BPE_pickle"
TRAINED_BPE_JSON = TRAINED_DATA_FOLDER + "/" + RAW_TEXT_NAME.split(".")[0] + "_trained_BPE_json"


SPECIAL_TOKENS = ["<|endoftext|>"]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer(ABC):

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@dataclass
class BPETokenizerParams():
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


def merge_tokens(vocab: dict[int, bytes], indices: list[int], pair: list[bytes, bytes], new_indice: int) -> list[int]:
    # Implement logic to merge two tokens into a new token
    new_indices = []
    n = len(indices)
    i = 0
    while i < n:
        if i+1 < n and vocab[indices[i]] == pair[0] and vocab[indices[i+1]] == pair[1]:
            new_indices.append(new_indice)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPE(Tokenizer):
    def __init__(self, params:BPETokenizerParams):
        self.params = params
        self.reverse_vocab = {v: k for k, v in self.params.vocab.items()}

    def encode(self, string: str) -> list[int]:
        # Implement BPE encoding logic here
        indices = list(map(int, string.encode("utf-8")))
        for pair in self.params.merges:
            new_indice = 256 + self.reverse_vocab[pair]
            indices = merge_tokens(self.params.vocab, indices, pair, new_indice)
        return indices
        

    def decode(self, indices: list[int]) -> str:
        # Implement BPE decoding logic here
        bytes_list = []
        for indice in indices:
            bytes_list.append(self.params.vocab.get(indice))
        string = b"".join(bytes_list).decode("utf-8")
        return string
    

def counts_pairs_update(words_tokens: dict[str, list[int]],
                        counts_words: dict[str, int],
                        pairs_to_words: dict[tuple[bytes, bytes], set[str]], 
                        counts_pairs: dict[tuple[bytes, bytes], int], 
                        vocab: dict[int, bytes],
                        pair: tuple[bytes, bytes],
                        new_indice: int):
    del counts_pairs[pair]

    words_list = deepcopy(pairs_to_words[pair])
    pair0_bytes, pair1_bytes = pair
    # pair0, pair1 = int(pair0_bytes), int(pair1_bytes)
    new_bytes = pair0_bytes + pair1_bytes
    for word in words_list:
        tokens = words_tokens[word]
        new_tokens = []
        i = 0
        while i < len(tokens)-1:
            if vocab[tokens[i]] == pair0_bytes and vocab[tokens[i+1]] == pair1_bytes:
                if i > 0:
                    remove_pair = tuple([vocab[tokens[i-1]], pair0_bytes])
                    if remove_pair in counts_pairs:
                        counts_pairs[remove_pair] -= counts_words[word]
                    add_pair = tuple([vocab[tokens[i-1]], new_bytes])
                    counts_pairs[add_pair] += counts_words[word]
                    pairs_to_words[remove_pair].discard(word)
                    pairs_to_words[add_pair].add(word)
                    
                if i < len(tokens)-2:
                    remove_pair = tuple([pair1_bytes, vocab[tokens[i+2]]])
                    if remove_pair in counts_pairs:
                        counts_pairs[remove_pair] -= counts_words[word]
                    add_pair = tuple([new_bytes, vocab[tokens[i+2]]])
                    counts_pairs[add_pair] += counts_words[word] 
                    pairs_to_words[remove_pair].discard(word)
                    pairs_to_words[add_pair].add(word)
                tokens[i+1] = new_indice
                new_tokens.append(new_indice)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if i == len(tokens)-1:
            new_tokens.append(tokens[i])
        words_tokens[word] = new_tokens
        # Re-add word to pairs_to_words for all pairs that still exist in new_tokens.
        # This corrects premature discards: when processing one occurrence of a pair
        # (e.g. (o,n) in 'condition'), the discard removes the word from
        # pairs_to_words for that pair even if another occurrence remains elsewhere.
        for j in range(len(new_tokens) - 1):
            remaining = (vocab[new_tokens[j]], vocab[new_tokens[j + 1]])
            pairs_to_words[remaining].add(word)                   


def BPE_tokenizer_training(input_path, vocab_size, special_tokens):
    counts_pairs = defaultdict(int)
    pairs_to_words = defaultdict(set)
    words_tokens = defaultdict(list)
    counts_words = Counter()
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pattern = "|".join(re.escape(tok) for tok in SPECIAL_TOKENS)
            splitted_chunks = re.split(pattern, chunk)
            logger.info(f"number of splitted chunks: {len(splitted_chunks)}")

            for splitted_chunk in splitted_chunks:
                # counts_words += Counter(splitted_chunk.split(" "))
                counts_words += Counter(re.findall(PAT, splitted_chunk))

        for key, value in counts_words.items():
            key_bytes = list(key.encode("utf-8"))
            words_tokens[key] = key_bytes
            for i in range(len(key_bytes)-1):
                pairs = (bytes([key_bytes[i]]), bytes([key_bytes[i+1]]))
                counts_pairs[pairs] += value
                pairs_to_words[pairs].add(key)
        logger.info(f"number of pairs: {len(counts_pairs)}")
        f.seek(0)
        indices = list(map(int, f.read()))


    assert vocab_size >= 256, "vocab should be >= 256"
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    for i in range(vocab_size - 256 - len(special_tokens)):
        if i > 0:
            counts_pairs_update(words_tokens, counts_words, pairs_to_words, counts_pairs, vocab, pair, new_indice)
        pair = max(counts_pairs, key= lambda x: (counts_pairs[x], x))
        new_indice = 256 + len(special_tokens) + i
        indices = merge_tokens(vocab, indices, pair, new_indice)
        vocab[new_indice] = pair[0] + pair[1]
        merges.append(pair)
        logger.info(f"iteration {i}, pair: {pair}, count: {counts_pairs[pair]}")

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    # input_path = RAW_TEXT_PATH
    input_path = "/Users/xshi849/Documents/playground/cs336-assignment1-Language-Modeling-From-Scratch/tests/fixtures/corpus.en"
    vocab_size = 500
    special_tokens = SPECIAL_TOKENS
    BPE_params = BPE_tokenizer_training(input_path, vocab_size, special_tokens)
    
    BPE_params_data = {
        "vocab": {str(k): base64.b64encode(v).decode("ascii") for k, v in BPE_params.vocab.items()},
        "merges": [[base64.b64encode(a).decode("ascii"), base64.b64encode(b).decode("ascii")] for a, b in BPE_params.merges]
    }

    if not os.path.isdir(TRAINED_DATA_FOLDER):
        os.mkdir(TRAINED_DATA_FOLDER)
    with open(f"{TRAINED_BPE_PICKLE}", "wb") as f:
        pickle.dump(BPE_params, f)
    
    with open(f"{TRAINED_BPE_JSON}", "w") as f:
        json.dump(BPE_params_data, f)