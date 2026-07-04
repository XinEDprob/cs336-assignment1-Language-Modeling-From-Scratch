import io
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


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _is_s3(path: str) -> bool:
    """Return True if *path* is an S3 URI (``s3://bucket/key``)."""
    return path.startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse ``s3://bucket/key/path`` into ``("bucket", "key/path")``."""
    without_scheme = uri[5:]          # strip "s3://"
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def find_chunk_boundaries_s3(
    bucket: str,
    key: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """S3 equivalent of find_chunk_boundaries using byte-range GET requests.

    Performs O(desired_num_chunks) tiny (4 KB) range requests to locate
    split-token boundaries without downloading the full file.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 support.  Run: pip install boto3")

    s3 = boto3.client("s3")
    file_size: int = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        pos = chunk_boundaries[bi]
        while pos < file_size:
            end_byte = min(pos + mini_chunk_size - 1, file_size - 1)
            resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={pos}-{end_byte}")
            mini_chunk: bytes = resp["Body"].read()
            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = pos + found_at
                break
            pos += len(mini_chunk)
        else:
            chunk_boundaries[bi] = file_size

    return sorted(set(chunk_boundaries))


def _count_words_in_chunk_s3(args: tuple) -> Counter:
    """Worker: count pre-token frequencies in one S3 file chunk.

    Issues a single byte-range GET so no worker ever loads the full file.
    Defined at module level so multiprocessing can pickle it on all platforms.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 support.  Run: pip install boto3")

    bucket, key, start, end, special_tokens_pattern, pat = args
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end - 1}")
    chunk = resp["Body"].read().decode("utf-8", errors="ignore")
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

    if _is_s3(input_path):
        s3_bucket, s3_key = _parse_s3_uri(input_path)
        boundaries = find_chunk_boundaries_s3(s3_bucket, s3_key, num_processes * 4, split_token)
        chunk_args = [
            (s3_bucket, s3_key, start, end, special_tokens_pattern, PAT)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        worker_fn = _count_words_in_chunk_s3
    else:
        with open(input_path, "rb") as f:
            # Use more chunks than processes so slow chunks don't stall the pool.
            boundaries = find_chunk_boundaries(f, num_processes * 4, split_token)
        chunk_args = [
            (input_path, start, end, special_tokens_pattern, PAT)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        worker_fn = _count_words_in_chunk

    counts_words: Counter = Counter()
    with multiprocessing.Pool(num_processes) as pool:
        # imap_unordered yields results as each worker finishes, so only one
        # partial Counter is held in the main process at a time instead of all
        # num_chunks Counters simultaneously (saves ~10 GB for 11 GB corpora).
        for partial_counts in pool.imap_unordered(worker_fn, chunk_args):
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


def save_bpe_params(params: BPETokenizerParams, output_dir: str, dataset_name: str) -> None:
    """Write *params* to *output_dir* (local directory **or** ``s3://bucket/prefix/``).

    Three files are written regardless of destination:

    * ``<dataset_name>_bpe.pkl``    – full params as a pickle
    * ``<dataset_name>_vocab.json`` – human-readable id → token string mapping
    * ``<dataset_name>_merges.txt`` – GPT-2 style merge list
    """
    vocab, merges = params.vocab, params.merges

    # Serialise into bytes first so the same code path works for both local
    # writes and S3 uploads.
    pkl_bytes = pickle.dumps(params)

    vocab_readable = {
        str(token_id): tok.decode("utf-8", "replace")
        for token_id, tok in vocab.items()
    }
    vocab_json_bytes = json.dumps(vocab_readable, ensure_ascii=False, indent=2).encode("utf-8")

    merge_lines = ["#version: 1.0\n"] + [
        f"{a.decode('utf-8', 'replace')} {b.decode('utf-8', 'replace')}\n"
        for a, b in merges
    ]
    merges_bytes = "".join(merge_lines).encode("utf-8")

    files: dict[str, bytes] = {
        f"{dataset_name}_bpe.pkl":    pkl_bytes,
        f"{dataset_name}_vocab.json": vocab_json_bytes,
        f"{dataset_name}_merges.txt": merges_bytes,
    }

    if _is_s3(output_dir):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 support.  Run: pip install boto3")
        s3 = boto3.client("s3")
        bucket, prefix = _parse_s3_uri(output_dir)
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        for filename, data in files.items():
            s3_key = prefix + filename
            s3.upload_fileobj(io.BytesIO(data), bucket, s3_key)
            logger.info(f"Saved s3://{bucket}/{s3_key}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        for filename, data in files.items():
            local_path = os.path.join(output_dir, filename)
            with open(local_path, "wb") as f:
                f.write(data)
            logger.info(f"Saved {local_path}")


if __name__ == "__main__":
    # Both local paths and S3 URIs are accepted, e.g.:
    #   input_path = "s3://my-bucket/data/owt_train.txt"
    #   output_dir = "s3://my-bucket/trained/"
    # The EC2 instance must have an IAM role (or ~/.aws/credentials) with
    # s3:GetObject / s3:PutObject permissions on the relevant bucket.
    input_path = RAW_TEXT_PATH
    output_dir = TRAINED_DATA_FOLDER

    DATASET_NAME = input_path.rstrip("/").split("/")[-1].split(".")[0]
    vocab_size = 10000
    special_tokens = SPECIAL_TOKENS

    BPE_params = BPE_tokenizer_training(input_path, vocab_size, special_tokens)
    save_bpe_params(BPE_params, output_dir, DATASET_NAME)
    print("BPE tokenizer training completed and saved.")