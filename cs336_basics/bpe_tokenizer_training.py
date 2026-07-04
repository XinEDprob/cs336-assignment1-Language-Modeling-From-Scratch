import os
import regex as re
import logging
from typing import BinaryIO
from collections import Counter, defaultdict
from dataclasses import dataclass
import pickle
import json
import heapq
import multiprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


RAW_TEXT_FOLDER_PATH = "/Users/xshi849/Documents/playground/cs336-assignment1-Language-Modeling-From-Scratch/data"
# RAW_TEXT_NAME = "TinyStoriesV2-GPT4-train.txt"
RAW_TEXT_NAME = "owt_valid.txt"
RAW_TEXT_PATH = RAW_TEXT_FOLDER_PATH + "/" + RAW_TEXT_NAME

TRAINED_DATA_FOLDER = "/Users/xshi849/Documents/playground/cs336-assignment1-Language-Modeling-From-Scratch/trained"
TRAINED_BPE_PICKLE = TRAINED_DATA_FOLDER + "/" + RAW_TEXT_NAME.split(".")[0] + "_trained_BPE_pickle"
TRAINED_BPE_JSON = TRAINED_DATA_FOLDER + "/" + RAW_TEXT_NAME.split(".")[0] + "_trained_BPE_json"


SPECIAL_TOKENS = ["<|endoftext|>"]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def _count_words_in_chunk(args: tuple) -> Counter:
    """Worker: count pre-token frequencies in one file chunk.

    Defined at module level so multiprocessing can pickle it on all platforms.
    """
    input_path, start, end, special_tokens_pattern, pat = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    local_counts: Counter = Counter()
    for sub in re.split(special_tokens_pattern, chunk):
        local_counts.update(re.findall(pat, sub))
    return local_counts


@dataclass
class BPETokenizerParams():
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


def merge_tokens(vocab: dict[int, bytes], indices: list[int], pair: tuple[bytes, bytes], new_indice: int) -> list[int]:
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


def counts_pairs_update(
        words_tokens: dict[str, list[int]],
        counts_words: dict[str, int],
        pairs_to_words: dict[tuple[bytes, bytes], set[str]],
        counts_pairs: dict[tuple[bytes, bytes], int],
        heap: list,
        vocab: dict[int, bytes],
        pair: tuple[bytes, bytes],
        new_indice: int,
) -> None:
    """Merge *pair* into *new_indice* across all affected words, then update every
    data structure (counts_pairs, pairs_to_words, words_tokens) and push fresh
    heap entries for modified pairs (lazy-deletion heap; stale entries are
    ignored when popped).
    """
    pair0_bytes, pair1_bytes = pair
    new_bytes = pair0_bytes + pair1_bytes

    # Shallow copy is sufficient – we only need a snapshot of the word set.
    words_list = set(pairs_to_words[pair])

    # Remove the now-merged pair from all tracking structures.
    del counts_pairs[pair]
    del pairs_to_words[pair]

    for word in words_list:
        tokens = words_tokens[word]
        freq = counts_words[word]
        new_tokens: list[int] = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and vocab[tokens[i]] == pair0_bytes
                and vocab[tokens[i + 1]] == pair1_bytes
            ):
                # ── left neighbour ──────────────────────────────────────────
                # Use new_tokens[-1] (not tokens[i-1]) so that cascaded merges
                # within the same word are handled correctly (e.g. "aaaa" → AA AA
                # requires the second merge's left context to be AA, not a).
                if new_tokens:
                    left_bytes = vocab[new_tokens[-1]]

                    remove_pair = (left_bytes, pair0_bytes)
                    if remove_pair in counts_pairs:
                        counts_pairs[remove_pair] -= freq
                        if counts_pairs[remove_pair] <= 0:
                            del counts_pairs[remove_pair]
                        else:
                            heapq.heappush(heap, (-counts_pairs[remove_pair], remove_pair))
                    pairs_to_words[remove_pair].discard(word)

                    add_pair = (left_bytes, new_bytes)
                    counts_pairs[add_pair] += freq
                    heapq.heappush(heap, (-counts_pairs[add_pair], add_pair))
                    pairs_to_words[add_pair].add(word)

                # ── right neighbour ─────────────────────────────────────────
                if i + 2 < len(tokens):
                    right_bytes = vocab[tokens[i + 2]]

                    remove_pair = (pair1_bytes, right_bytes)
                    if remove_pair in counts_pairs:
                        counts_pairs[remove_pair] -= freq
                        if counts_pairs[remove_pair] <= 0:
                            del counts_pairs[remove_pair]
                        else:
                            heapq.heappush(heap, (-counts_pairs[remove_pair], remove_pair))
                    pairs_to_words[remove_pair].discard(word)

                    add_pair = (new_bytes, right_bytes)
                    counts_pairs[add_pair] += freq
                    heapq.heappush(heap, (-counts_pairs[add_pair], add_pair))
                    pairs_to_words[add_pair].add(word)

                new_tokens.append(new_indice)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        words_tokens[word] = new_tokens

        # Re-add word for every pair still present in new_tokens.  This corrects
        # premature discards: when one occurrence of a pair is processed the word
        # is discarded from pairs_to_words even if another occurrence remains.
        for j in range(len(new_tokens) - 1):
            remaining = (vocab[new_tokens[j]], vocab[new_tokens[j + 1]])
            pairs_to_words[remaining].add(word)


def BPE_tokenizer_training(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:
    """Train a BPE tokenizer on *input_path*.

    Designed to handle gigabyte-scale corpora:
    * Pre-tokenisation is parallelised across all CPU cores.
    * Pair selection uses an O(log n) max-heap with lazy deletion instead of
      an O(n) linear scan per merge step.
    * Only word-frequency statistics are kept in memory (not the full token
      sequence), so memory is proportional to vocabulary size, not file size.
    """
    num_processes = os.cpu_count() or 4
    # Pattern that splits text on special tokens (used inside each worker too)
    special_tokens_pattern = "|".join(re.escape(tok) for tok in special_tokens) if special_tokens else "$^"  # $^ never matches

    # ── Step 1: count pre-token frequencies in parallel ───────────────────────
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    with open(input_path, "rb") as f:
        # Use more chunks than processes so slow chunks don't stall the pool.
        boundaries = find_chunk_boundaries(f, num_processes * 4, split_token)

    chunk_args = [
        (input_path, start, end, special_tokens_pattern, PAT)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    counts_words: Counter = Counter()
    with multiprocessing.Pool(num_processes) as pool:
        # imap_unordered yields results as each worker finishes, so only one
        # partial Counter is held in the main process at a time instead of all
        # num_chunks Counters simultaneously (saves ~10 GB for 11 GB corpora).
        for partial_counts in pool.imap_unordered(_count_words_in_chunk, chunk_args):
            counts_words.update(partial_counts)

    logger.info(f"Unique pre-tokens: {len(counts_words):,}")

    # ── Step 2: build initial pair-count data structures ─────────────────────
    counts_pairs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pairs_to_words: dict[tuple[bytes, bytes], set[str]] = defaultdict(set)
    words_tokens: dict[str, list[int]] = {}

    for word, freq in counts_words.items():
        token_ids = list(word.encode("utf-8"))
        words_tokens[word] = token_ids
        for i in range(len(token_ids) - 1):
            pair = (bytes([token_ids[i]]), bytes([token_ids[i + 1]]))
            counts_pairs[pair] += freq
            pairs_to_words[pair].add(word)

    logger.info(f"Unique initial pairs: {len(counts_pairs):,}")

    # ── Step 3: max-heap for O(log n) best-pair selection (lazy deletion) ─────
    # Stored as (-count, pair) so heapq (a min-heap) gives us the max-count pair.
    heap: list[tuple[int, tuple[bytes, bytes]]] = [
        (-count, pair) for pair, count in counts_pairs.items()
    ]
    heapq.heapify(heap)

    # ── Step 4: initialise vocabulary ────────────────────────────────────────
    assert vocab_size >= 256, "vocab_size must be >= 256"
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - 256 - len(special_tokens)

    # ── Step 5: BPE merge loop ────────────────────────────────────────────────
    for i in range(num_merges):
        # Pop from heap until we find an entry that still reflects the true count
        # (lazy deletion: stale entries for pairs whose counts changed are skipped).
        pair: tuple[bytes, bytes] | None = None
        while heap:
            neg_count, candidate = heapq.heappop(heap)
            if candidate in counts_pairs and counts_pairs[candidate] == -neg_count:
                pair = candidate
                break

        if pair is None:
            logger.warning(f"No pairs remain after {i} merges; stopping early.")
            break

        new_id = 256 + len(special_tokens) + i
        vocab[new_id] = pair[0] + pair[1]
        merges.append(pair)
        logger.info(f"Merge {i + 1}/{num_merges}: {pair!r}  count={counts_pairs[pair]:,}")

        counts_pairs_update(
            words_tokens, counts_words, pairs_to_words, counts_pairs, heap, vocab, pair, new_id
        )

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    input_path = RAW_TEXT_PATH
    # input_path = "/Users/xshi849/Documents/playground/cs336-assignment1-Language-Modeling-From-Scratch/tests/fixtures/corpus.en"
    DATASET_NAME = input_path.split("/")[-1].split(".")[0]
    vocab_size = 10000
    special_tokens = SPECIAL_TOKENS
    BPE_params = BPE_tokenizer_training(input_path, vocab_size, special_tokens)
    
    vocab = BPE_params.vocab
    merges = BPE_params.merges

    if not os.path.isdir(TRAINED_DATA_FOLDER):
        os.mkdir(TRAINED_DATA_FOLDER)
    with open(f"{TRAINED_DATA_FOLDER}/{DATASET_NAME}_bpe.pkl", "wb") as f:
        pickle.dump(BPE_params, f)

    # Human-readable vocab: token string → id
    vocab_readable = {id: token.decode("utf-8", "replace") for id, token in vocab.items()}
    with open(f"{TRAINED_DATA_FOLDER}/{DATASET_NAME}_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_readable, f, ensure_ascii=False, indent=2)

    # Human-readable merges: GPT-2 style, one merge per line
    with open(f"{TRAINED_DATA_FOLDER}/{DATASET_NAME}_merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 1.0\n")
        for a, b in merges:
            f.write(f"{a.decode('utf-8', 'replace')} {b.decode('utf-8', 'replace')}\n")

    # NOTE: For large corpora, use BPE_Tokenizer.encode() from bpe_tokenizer.py
    # instead of re-tokenising here.  The naive loop below is O(vocab_size × tokens)
    # and is impractical for GB-scale data.

    print("BPE tokenizer training completed and saved to disk.")