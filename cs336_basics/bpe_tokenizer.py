import io
import os
import logging
from typing import Iterator
from cs336_basics.bpe_tokenizer_training import SPECIAL_TOKENS, PAT, merge_tokens, _is_s3, _parse_s3_uri, BPETokenizerParams
from abc import ABC, abstractmethod
import pickle
import regex as re
import json


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class _BPEParamsUnpickler(pickle.Unpickler):
    """Remaps pickles saved when bpe_tokenizer_training.py was run as __main__.

    When the training script is executed directly (``python bpe_tokenizer_training.py``),
    Python records the class as ``__main__.BPETokenizerParams`` in the pickle.
    Loading that pickle from any other __main__ fails because the class is not
    found there.  This unpickler transparently redirects the lookup to the
    canonical module path regardless of how the training script was invoked.
    """
    def find_class(self, module: str, name: str):
        if name == "BPETokenizerParams":
            return BPETokenizerParams
        return super().find_class(module, name)


class Tokenizer(ABC):

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class BPE_Tokenizer(Tokenizer):
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = special_tokens if special_tokens is not None else []
        # Pre-compute merged token IDs (avoids repeated reverse-vocab lookups)
        self._merge_ids = [self.reverse_vocab[p[0] + p[1]] for p in self.merges]
        # Pre-compile the special-token split pattern (avoids rebuilding on every encode)
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern: str | None = "|".join(re.escape(t) for t in sorted_special)
        else:
            self._special_pattern = None
        self._special_token_set: set[str] = set(self.special_tokens)
        # Word-level BPE cache: pre-token string → final list of token IDs.
        # Each unique word type is merged only once, no matter how many times
        # it appears in the text.  For a 1 GB corpus with ~100 K unique word
        # types this eliminates ~2,500× redundant merge work.
        self._word_cache: dict[str, list[int]] = {}
        # Merge-rank lookup: pair of bytes → index in self.merges (lower = higher priority).
        # Enables O(1) pair-priority lookup inside _encode_word.
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

    def _encode_word(self, word: str) -> list[int]:
        """BPE-encode a single pre-token, caching the result for reuse.

        Uses a priority-based merge loop: find the applicable merge with the
        lowest rank (highest priority), apply it, repeat.  This is O(n²) in
        word length n rather than the previous O(n × num_merges), which gives
        a ~2,000× speedup for typical 4-6 byte words with 10 K+ merges.
        """
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        ids = [self.reverse_vocab[bytes([b])] for b in word.encode("utf-8")]
        num_merges = len(self.merges)

        while len(ids) > 1:
            # Scan all adjacent pairs and find the one with the lowest merge rank.
            best_rank = num_merges  # sentinel – means "not found"
            best_i = -1
            for i in range(len(ids) - 1):
                rank = self._merge_rank.get(
                    (self.vocab[ids[i]], self.vocab[ids[i + 1]]), num_merges
                )
                if rank < best_rank:
                    best_rank = rank
                    best_i = i

            if best_i == -1:
                break  # no applicable merge remains

            # Apply the merge in-place (no new list allocation).
            ids[best_i] = self._merge_ids[best_rank]
            del ids[best_i + 1]

        self._word_cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        chunks = re.split(f"({self._special_pattern})", text) if self._special_pattern else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self._special_token_set:
                token_ids.append(self.reverse_vocab[chunk.encode("utf-8")])
            else:
                for word in re.findall(PAT, chunk):
                    token_ids.extend(self._encode_word(word))
        return token_ids

    def decode(self, indices: list[int]) -> str:
        return b"".join(self.vocab[i] for i in indices).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def encode_file(
        self,
        input_path: str,
        output_path: str,
        write_batch: int = 1_000_000,
        log_every_mb: float = 100.0,
    ) -> int:
        """Stream-encode a large text file to a flat binary token-ID array.

        Reads and encodes the file in batches so only a small constant amount
        of memory is used at a time.  The word cache persists across batches,
        so each unique word type is BPE-merged exactly once regardless of how
        many times it appears in the file.

        Parameters
        ----------
        input_path : str
            Local path or ``s3://bucket/key``.
        output_path : str
            Local path or ``s3://bucket/key``.
            Written as a flat raw binary array (uint16 for vocab < 65 536,
            uint32 otherwise).  Reload with::

                import numpy as np
                tokens = np.fromfile(output_path, dtype=np.uint16)
        write_batch : int
            Number of token IDs to buffer before flushing to disk.
        log_every_mb : float
            Emit a progress log line roughly every this many MB of input read.
            Set to 0 to disable mid-run logging.

        Returns
        -------
        int
            Total number of tokens written.
        """
        import numpy as np
        import time

        max_id = max(self.vocab.keys())
        np_dtype = np.uint16 if max_id < 65_536 else np.uint32

        # Total file size for ETA (local only; S3 would need an extra HeadObject)
        file_size: int | None = None
        if not _is_s3(input_path):
            try:
                file_size = os.path.getsize(input_path)
            except OSError:
                pass

        log_interval = int(log_every_mb * 1_000_000) if log_every_mb > 0 else 0

        def _iter_lines(path: str):
            """Yield text lines from a local path or S3 URI."""
            if _is_s3(path):
                try:
                    import boto3
                except ImportError:
                    raise ImportError("boto3 is required for S3 support.  Run: pip install boto3")
                bucket, key = _parse_s3_uri(path)
                body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]
                for raw in body.iter_lines():
                    yield raw.decode("utf-8", errors="ignore") + "\n"
            else:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    yield from f

        def _encode_and_write(out_f) -> int:
            total_tokens = 0
            bytes_read = 0
            last_log_at = 0
            t_start = time.perf_counter()
            batch: list[int] = []

            for line in _iter_lines(input_path):
                batch.extend(self.encode(line))
                bytes_read += len(line.encode("utf-8", errors="ignore"))

                if len(batch) >= write_batch:
                    np.array(batch, dtype=np_dtype).tofile(out_f)
                    total_tokens += len(batch)
                    batch = []

                if log_interval and bytes_read - last_log_at >= log_interval:
                    elapsed = time.perf_counter() - t_start
                    mb_s = bytes_read / elapsed / 1e6 if elapsed > 0 else 0.0
                    if file_size:
                        pct = bytes_read / file_size * 100
                        eta = (file_size - bytes_read) / (bytes_read / elapsed) if bytes_read > 0 else 0.0
                        logger.info(
                            f"[encode_file] {bytes_read/1e6:,.0f} / {file_size/1e6:,.0f} MB"
                            f"  ({pct:.1f}%)  {total_tokens/1e6:.1f}M tokens"
                            f"  {mb_s:.1f} MB/s  ETA {eta:.0f}s"
                        )
                    else:
                        logger.info(
                            f"[encode_file] {bytes_read/1e6:,.0f} MB"
                            f"  {total_tokens/1e6:.1f}M tokens  {mb_s:.1f} MB/s"
                        )
                    last_log_at = bytes_read

            if batch:
                np.array(batch, dtype=np_dtype).tofile(out_f)
                total_tokens += len(batch)

            return total_tokens

        if _is_s3(output_path):
            try:
                import boto3, tempfile
            except ImportError:
                raise ImportError("boto3 is required for S3 support.  Run: pip install boto3")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                tmp_path = tmp.name
            try:
                with open(tmp_path, "wb") as out_f:
                    total_tokens = _encode_and_write(out_f)
                bucket, key = _parse_s3_uri(output_path)
                boto3.client("s3").upload_file(tmp_path, bucket, key)
            finally:
                os.remove(tmp_path)
        else:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(out_dir, exist_ok=True)
            with open(output_path, "wb") as out_f:
                total_tokens = _encode_and_write(out_f)

        elapsed = time.perf_counter()  # not accurate here, but final summary is below
        logger.info(
            f"[encode_file] done — {total_tokens:,} tokens"
            f" ({np_dtype.__name__}, {total_tokens * np.dtype(np_dtype).itemsize / 1e9:.2f} GB)"
            f" → {output_path}"
        )
        return total_tokens

    @classmethod
    def from_files(cls, BPE_params_filepath, special_tokens=None):
        if _is_s3(BPE_params_filepath):
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 is required for S3 support.  Run: pip install boto3")
            bucket, key = _parse_s3_uri(BPE_params_filepath)
            s3 = boto3.client("s3")
            data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            BPE_params = _BPEParamsUnpickler(io.BytesIO(data)).load()
        else:
            with open(BPE_params_filepath, "rb") as f:
                BPE_params = _BPEParamsUnpickler(f).load()
        return cls(BPE_params.vocab, BPE_params.merges, special_tokens)
    

    @classmethod
    def from_files_text(cls, vocab_filepath, merges_filepath, special_tokens=None):
        def _read_text(path: str) -> str:
            if _is_s3(path):
                try:
                    import boto3
                except ImportError:
                    raise ImportError("boto3 is required for S3 support.  Run: pip install boto3")
                bucket, key = _parse_s3_uri(path)
                s3 = boto3.client("s3")
                return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            with open(path, encoding="utf-8") as f:
                return f.read()

        vocab = json.loads(_read_text(vocab_filepath))
        merges = [
            tuple(line.rstrip().split(" "))
            for line in _read_text(merges_filepath).splitlines()
        ]
        return cls(vocab, merges, special_tokens)

    
if __name__ == "__main__":
    # ── small in-memory example ───────────────────────────────────────────────
    TRAINED_BPE_PARAMS_PATH = "/Users/xshi849/Documents/playground/cs336-assignment1-Language-Modeling-From-Scratch/trained/TinyStoriesV2-GPT4-train_bpe.pkl"
    tokenizer = BPE_Tokenizer.from_files(TRAINED_BPE_PARAMS_PATH, special_tokens=SPECIAL_TOKENS)
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")

    # ── GB-scale file example (local or S3) ───────────────────────────────────
    # For large files use encode_file instead of encode(text).
    # encode_file streams line-by-line (constant RAM) and writes a flat binary
    # numpy array that can be memory-mapped for training.
    #
    # Local example:
    #   n = tokenizer.encode_file(
    #       input_path  = "/data/owt_train.txt",
    #       output_path = "/data/owt_train_tokens.bin",
    #   )
    #
    # S3 example:
    #   n = tokenizer.encode_file(
    #       input_path  = "s3://my-bucket/data/owt_train.txt",
    #       output_path = "s3://my-bucket/data/owt_train_tokens.bin",
    #   )
    #
    # Reload the result:
    #   import numpy as np
    #   tokens = np.fromfile("owt_train_tokens.bin", dtype=np.uint16)

