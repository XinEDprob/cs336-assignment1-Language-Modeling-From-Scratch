import os
import re
import logging
from typing import BinaryIO
from collections import Counter, defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import json
import base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


RAW_TEXT_FOLDER_PATH = "/Users/xinshi/Documents_local/cs336/cs336-assignment1-Language-Modeling-From-Scratch/data"
RAW_TEXT_NAME = "TinyStoriesV2-GPT4-valid.txt"
RAW_TEXT_PATH = RAW_TEXT_FOLDER_PATH + "/" + RAW_TEXT_NAME

TRAINED_DATA_FOLDER = "/Users/xinshi/Documents_local/cs336/cs336-assignment1-Language-Modeling-From-Scratch/trained"
TRAINED_BPE_PICKLE = TRAINED_DATA_FOLDER + "/" + RAW_TEXT_NAME.split(".")[0] + "_trained_BPE_pickle"
TRAINED_BPE_JSON = TRAINED_DATA_FOLDER + "/" + RAW_TEXT_NAME.split(".")[0] + "_trained_BPE_json"


SPECIAL_TOKENS = ["<|endoftext|>"]


class Tokenizer(ABC):

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        return NotImplementedError
    
    @abstractmethod
    def decode(self, string: str) -> list[int]:
        return NotImplementedError


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


def merge_tokens(indices: list[int], pair: list[int, int], new_indice: int) -> list[int]:
    # Implement logic to merge two tokens into a new token
    new_indices = []
    n = len(indices)
    i = 0
    while i < n-1:
        if i+1 < n and indices[i] == pair[0] and indices[i+1] == pair[1]:
            i += 2
            new_indices.append(new_indice)
        else:
            i += 1
            new_indices.append(indices[i])
    return new_indices


class BPE(Tokenizer):
    def __init__(self, params:BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        # Implement BPE encoding logic here
        indices = list(map(int, string.encode("utf-8")))
        for pair, new_indice in self.params.merges.items():
            indices = merge_tokens(indices, pair, new_indice)
        return indices
        

    def decode(self, indices: list[int]) -> str:
        # Implement BPE decoding logic here
        bytes_list = []
        for indice in indices:
            bytes_list.append(self.params.vocab.get(indice))
        string = b"".join(bytes_list).decode("uft-8")
        return string
    

def counts_pairs_update(counts_pairs: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    raise NotImplementedError


def BPE_tokenizer_training(input_path, vocab_size, special_tokens):
    counts_pairs = defaultdict(int)

    ## Usage
    with open(RAW_TEXT_PATH, "rb") as f:
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

            counts_words = Counter()
            for splitted_chunk in splitted_chunks:
                counts_words += Counter(splitted_chunk.split(" "))

            for key, value in counts_words.items():
                for i in range(len(key)-1):
                    counts_pairs[key[i:i+2].encode("utf-8")] += value
            logger.info(f"number of pairs: {len(counts_pairs)}")
        # TODO check implementation details???
        f.seek(0)
        indices = list(map(int, f.read()))


    assert vocab_size >= 256, "vocab should be >= 256"
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    for i in range(vocab_size - 256):
        if i > 0:
            counts_pairs_update(counts_pairs)
        pair = max(counts_pairs, key=counts_pairs.get)
        new_indice = 256 + i
        indices = merge_tokens(indices, pair, new_indice)
        vocab[new_indice] = vocab[pair[0]] + vocab[pair[1]]
        merges.append(pair)

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    input_path = RAW_TEXT_PATH
    vocab_size = 2560
    special_tokens = SPECIAL_TOKENS
    BPE_params = BPE_tokenizer_training(input_path, vocab_size, special_tokens)
    
    data = {
        "vocab": {str(k): base64.b64encode(v).decode("ascii") for k, v in BPE_params.vocab.items()},
        "merges": [[base64.b64encode(a).decode("ascii"), base64.b64encode(b).decode("ascii")] for a, b in BPE_params.merges]
    }

    if not os.path.isdir(TRAINED_DATA_FOLDER):
        os.mkdir(TRAINED_DATA_FOLDER)
    with open(f"{TRAINED_BPE_PICKLE}", "wb") as f:
        pickle.dump(BPE_params, f)
    
    # with open(f"{TRAINED_BPE_JSON}", "wb") as f:
    #     json.dump(BPE_params, f)

    print("end")