from collections import defaultdict, Counter
from typing import Any
import regex as re
import multiprocessing as mp
import seaborn as sns
from tqdm import tqdm

pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
pattern = re.compile(pattern)

# count words
def count_words(doc: Any) -> dict:
    return Counter([tuple(segment.encode('utf-8')) for segment in re.findall(pattern, doc.split("See also")[0])])

# count words in a list of documents
def count_words_in_documents(documents: list, processes: int = 1) -> dict:
    """
        - Count the frequency of each segment in a list of documents
    """
    word_counts = Counter()
    chunk_size = (len(documents) // processes) if len(documents) > processes else 1
    with mp.Pool(processes = processes) as pool:
        for counter in pool.map(count_words, documents, chunksize=chunk_size):
            word_counts.update(counter)  # Faster in-place update
    return word_counts

# count pairs of tokens
def count_pairs_in_chunk(chunk):
    """
    Process a chunk of word counts and return a single Counter
    """
    pairs = []
    for word_count in chunk:
        word, count = word_count
        pair_counts = [((p1, p2), count) for p1, p2 in zip(word[:-1], word[1:])]
        pairs.extend(pair_counts)
    return Counter(dict(pairs))

def count_pairs_in_word_counts(word_counts: dict, processes: int = 1, chunk_size: int = 13000) -> Counter:
    """
    Count pairs using multiprocessing with chunking for better performance
    
    Args:
        word_counts: Dictionary of word counts
        processes: Number of processes to use
        chunk_size: Size of chunks to process at once
    """
    items = list(word_counts.items())
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    with mp.Pool(processes) as pool:
        chunk_counters = list(tqdm(
            pool.imap(count_pairs_in_chunk, chunks),
            total=len(chunks),
            desc="Counting token pairs"
        ))
    final_counter = Counter()
    for counter in chunk_counters:
        final_counter.update(counter)
    return final_counter

def merge_pairs(word_count_arg: tuple) -> tuple:
    word_count, pair, new_tok = word_count_arg
    toks, freq = word_count
    new_tokens = []
    i = 0
    while i < len(toks):
        if i < len(toks) - 1 and (toks[i], toks[i + 1]) == pair:
            new_tokens.append(new_tok)
            i += 2
        else:
            new_tokens.append(toks[i])
            i += 1
    return (tuple(new_tokens), freq)

def merge_in_chunks(chunk_data) -> tuple[dict, dict]:
    chunk, pair, new_tok = chunk_data
    chunk_word_counts = defaultdict(int)
    pairs = Counter()
    for toks, freq in chunk:
        if pair not in zip(toks, toks[1:]):
            chunk_word_counts[toks] += freq
        else:
            new_toks, new_freq = merge_pairs(((toks, freq), pair, new_tok))
            chunk_word_counts[new_toks] += new_freq
            pairs += Counter(zip(new_toks, new_toks[1:]))
    return dict(chunk_word_counts), pairs

def merge_pairs_in_word_counts(word_counts: dict, pairs: Counter, pair: tuple[int, int], new_tok: int, processes: int = 1, chunk_size: int = 10000) -> tuple[dict, dict]:
    items = list(word_counts.items())
    def chunk_iterator():
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            yield (chunk, pair, new_tok)
    merged_word_counts = defaultdict(int)
    with mp.Pool(processes) as pool:
        for chunk_word_counts, chunk_new_pairs in pool.imap(merge_in_chunks, chunk_iterator()):
            for k, v in chunk_word_counts.items():
                merged_word_counts[k] += v
            pairs += chunk_new_pairs
    pairs[pair] = 0
    return dict(merged_word_counts), pairs

# @lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    credit - openai/gpt2
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
            
def plot_loss_curve(epochs,train_loss_values,test_loss_values):
    sns.lineplot(x=epochs,y=train_loss_values,label='Train Loss',color='green')
    if test_loss_values:
        sns.lineplot(x=epochs,y=test_loss_values)
