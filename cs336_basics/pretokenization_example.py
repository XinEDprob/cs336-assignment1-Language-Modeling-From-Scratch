import os
import re
import logging
from typing import BinaryIO
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


RAW_TEXT_PATH = "/Users/xinshi/Documents_local/cs336/cs336-assignment1-Language-Modeling-From-Scratch/data/TinyStoriesV2-GPT4-valid.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]

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
                counts_pairs[key[i:i+2]] += value
        logger.info(f"number of pairs: {len(counts_pairs)}")

    print("end")
