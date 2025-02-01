import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch import Tensor
import regex as re
from tqdm import tqdm
import json
import os
import tqdm
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import numpy as np
from collections import defaultdict, Counter
from typing import Any
import regex as re
import multiprocessing as mp
import seaborn as sns
from tqdm import tqdm

pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
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
        counters = pool.map(count_words, documents, chunksize=chunk_size)
    for counter in counters:
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
    sns.lineplot(x=epochs,y=train_loss_values)
    if test_loss_values:
        sns.lineplot(x=epochs,y=test_loss_values)

@dataclass
class ModelParams:
    """
    Configuration parameters for the transformer model.

    Attributes:
        context_length: Maximum sequence length for input tokens
        vocab_size: Number of unique tokens in the vocabulary
        num_blocks: Number of decoder blocks in the model
        num_heads: Number of attention heads
        d_model: Model's embedding dimension
        head_dim: Dimension of each attention head
        dropout_rate: Probability of dropout
        num_of_hidden_units: Number of units in feedforward hidden layer
        device: Computing device (cuda or cpu)
    """
    context_length: int = 512
    vocab_size: int = 50257
    num_blocks: int = 12
    num_heads: int = 12
    d_model: int = 768
    head_dim: int = field(init=False)
    dropout_rate: float = 0.1
    num_of_hidden_units: int = 3072
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Validate and compute head dimension after initialization."""
        assert self.d_model % self.num_heads == 0, "Number of heads must divide model dimension"
        self.head_dim = self.d_model // self.num_heads


class AttentionHead(nn.Module):
    """
    Single attention head for multi-head attention mechanism.

    Args:
        params: Model configuration parameters
    """
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.key = nn.Linear(params.d_model, params.head_dim)
        self.query = nn.Linear(params.d_model, params.head_dim)
        self.value = nn.Linear(params.d_model, params.head_dim)
        self.dropout = nn.Dropout(p=params.dropout_rate)
        self.device = params.device

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute scaled masked attention for a single head.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            Attention output tensor
        """
        k = self.dropout(self.key(x))
        q = self.dropout(self.query(x))
        v = self.dropout(self.value(x))
        _, T, dk = k.shape
        dot_product_attention = q @ k.transpose(2, 1)
        scaled_dot_product_attention = dot_product_attention / torch.sqrt(torch.tensor(dk))
        masked_attention = scaled_dot_product_attention.masked_fill(
            (torch.tril(torch.ones(T, T)) == 0).to(self.device), float("-inf")
        )
        soft_masked_attention = self.dropout(torch.softmax(masked_attention, dim=-1))
        return soft_masked_attention @ v


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module combining multiple attention heads.

    Args:
        params: Model configuration parameters
    """
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(params) for _ in range(params.num_heads)])
        self.proj = nn.Linear(params.d_model, params.d_model)
        self.dropout = nn.Dropout(p=params.dropout_rate)

    def forward(self, X: Tensor) -> Tensor:
        """
        Compute multi-head attention.

        Args:
            X: Input tensor

        Returns:
            Multi-head attention output
        """
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class PositionWiseFeedforward(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        params: Model configuration parameters
    """
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=params.d_model, out_features=params.num_of_hidden_units),
            nn.GELU(),
            nn.Linear(in_features=params.num_of_hidden_units, out_features=params.d_model),
            nn.Dropout(p=params.dropout_rate),
        )

    def forward(self, X: Tensor) -> Tensor:
        """
        Apply position-wise feed-forward transformation.

        Args:
            X: Input tensor

        Returns:
            Transformed tensor
        """
        return self.ffn(X)


class DecoderBlock(nn.Module):
    """
    Transformer decoder block with multi-head self-attention and feed-forward layers.

    Args:
        params: Model configuration parameters
    """
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.mmhsa = MultiHeadAttention(params)
        self.layer_norm1 = nn.LayerNorm(params.d_model)
        self.pwffn = PositionWiseFeedforward(params)
        self.layer_norm2 = nn.LayerNorm(params.d_model)

    def forward(self, X: Tensor) -> Tensor:
        """
        Process input through decoder block layers.

        Args:
            X: Input tensor

        Returns:
            Processed tensor
        """
        X = self.layer_norm1(X + self.mmhsa(X))
        out = self.layer_norm2(X + self.pwffn(X))
        return out


class TinyLLM(nn.Module):
    """
    Tiny Language Model implementation with transformer architecture.

    Args:
        params: Model configuration parameters
    """
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(num_embeddings=params.vocab_size, embedding_dim=params.d_model)
        self.position_embedding = nn.Embedding(num_embeddings=params.context_length, embedding_dim=params.d_model)
        self.embed_dropout = nn.Dropout(p=params.dropout_rate)
        self.blocks = nn.Sequential(*[DecoderBlock(params) for _ in range(params.num_blocks)])
        self.final_layer_norm = nn.LayerNorm(params.d_model)
        self.lm_head = nn.Linear(params.d_model, params.vocab_size)
        self.params: ModelParams = params
        
        # Weight tying between embedding and output layers
        self.text_embedding.weight = self.lm_head.weight

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass through the language model.

        Args:
            X: Input tensor of token indices

        Returns:
            Logits for next token predictions
        """
        _, T = X.shape
        text_embed = self.embed_dropout(self.text_embedding(X))
        pos_embed = self.embed_dropout(
            self.position_embedding(torch.arange(T, device=self.params.device))
        )
        X = text_embed + pos_embed
        H = self.blocks(X)
        H = self.final_layer_norm(X + H)
        logits = self.lm_head(H)
        return logits

    def generate(self, current_context: Tensor, max_new_tokens: int) -> Tensor:
        """
        Generate new tokens based on input context.

        Args:
            current_context: Initial token sequence
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated token sequence
        """
        for _ in range(max_new_tokens):
            current_context_cond = current_context[:, -self.params.context_length:]
            logits = self(current_context_cond)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_context = torch.cat((current_context, next_token), dim=1)
        return current_context
    
    def _num_parameters(self) -> int:
        """
        Calculate the number of trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
class TextEncoder:
    """
        Byte-pair encoding tokenizer.
    """
    def __init__(self):
        self.merges = {}
        self.vocab = self._build_vocab()
        self.special_tokens_list = ['<|endofdoc|>']    
        self.special_tokens = {}
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        self.pattern_compiled = re.compile(self.pattern)
        self.version = 0
        
    def _add_special_tokens_to_vocab(self):
        main_vocab_length = len(self.vocab)
        for i, token in enumerate(self.special_tokens_list):
            idx = main_vocab_length+i
            self.special_tokens[token] = idx
            self.vocab[idx] = token

    def _build_vocab(self):
        vocab = bytes_to_unicode()
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
    
    def train(self, word_counts: dict, vocab_size: int = 256, processes: int = 1):
        """
            Builds the merges for the tokenizer using the given list of wordcounts.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        initial_vocab_size = len(self.vocab)
        num_merges = vocab_size - initial_vocab_size
        pairs = count_pairs_in_word_counts(word_counts, processes)
        for i in tqdm(range(num_merges)):
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_tok = initial_vocab_size + i
            word_counts, pairs = merge_pairs_in_word_counts(word_counts, pairs = pairs, pair = best_pair,  new_tok = new_tok, processes = 30,  chunk_size = 100_000)
            self.merges[best_pair] = new_tok
            self.vocab[new_tok] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] # add the new token to the vocab assign it the value of the two merged tokens' binary values
        self.version += 1
        self._add_special_tokens_to_vocab()
        
    def _merge_tokens(self, tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        """
        Merges a pair of tokens in a list of tokens.
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
        
    def encode(self, text: str) -> list[int]:
        """
        Given a string, encodes it into a list of token IDs.
        """
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text) # returns a list of text separated by the special tokens with the tokens themselves in the list as well
        tokens = []
        for chunk in special_chunks:
            if chunk in self.special_tokens:
                tokens.append(self.special_tokens[chunk])
            else:
                text_chunks = re.findall(self.pattern_compiled, chunk)
                for part in text_chunks:
                    part_tokens = list(part.encode("utf-8")) # encode the part into bytes
                    while len(part_tokens) >= 2:
                        pairs = [pair for pair in zip(part_tokens, part_tokens[1:])]
                        pair = min(pairs, key=lambda p: self.merges.get(p,float("inf")))
                        if pair not in self.merges:
                            break
                        part_tokens = self._merge_tokens(part_tokens, pair, self.merges[pair])
                    tokens.extend(part_tokens)
        return tokens
    
    def decode(self, tokens: list[int]) -> str:
        """
        Given a list of token IDs, decodes them into a string.
        """
        inverse_special_tokens = {idx: st for st, idx in self.special_tokens.items()}
        inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        part_strs = []
        for idx in tokens:
            if idx in self.vocab:
                part_strs.append(self.vocab[idx])
            elif idx in inverse_special_tokens:
                part_strs.append(inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text = "".join(part_strs)
        text = bytearray([inverse_vocab[c] for c in text]).decode('utf-8', errors="replace")
        return text
    
    def save(self, path: str):
        """
            Saves the tokenizer to a file. Saves the merges, special tokens and pattern.
        """
        os.makedirs(path, exist_ok=True)
        merge_file, vocab_file = os.path.join(path, "tokenizer.merges"), os.path.join(path, "tokenizer.json")
        with open(merge_file, 'w', encoding='utf-8') as f:
            f.write(f"#dragontokenizer: {self.version}\n")
            f.write(f"#pattern: {self.pattern}\n")
            f.write(f"#special_tokens_count: {len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        vocab = {}
        for idx, token in self.vocab.items():
            if idx in inverted_merges:
                idx0, idx1 = inverted_merges[idx]
                s0 = self.vocab[idx0]
                s1 = self.vocab[idx1]
                vocab[s0+s1] = idx
            else:
                vocab[token] = idx
        vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f)
        
    @classmethod
    def load(cls, path: str):
        """Inverse of save() but only for the merges file"""
        model_file = os.path.join(path, "tokenizer.merges")
        merges, special_tokens, idx = {}, {}, 256
        tokenizer = cls()
        with open(model_file, 'r', encoding="utf-8") as f:
            version = int(f.readline().split(":")[1].strip())
            pattern = f.readline().strip()
            num_special = int(f.readline().split(":")[1].strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        tokenizer.version = version
        tokenizer.pattern = pattern
        tokenizer.merges = merges
        tokenizer.special_tokens = special_tokens
        tokenizer.vocab = tokenizer._build_vocab()
        tokenizer._add_special_tokens_to_vocab()
        return tokenizer


@dataclass
class TrainerParams:
    """
    Configuration parameters for the model trainer.

    Attributes:
        model: Language model to train
        train_data: Training data loader
        optimizer: Optimizer for training
        gpu_id: GPU device ID
        save_every: Frequency of saving model checkpoints
        loss_fn: Loss function for training
    """
    model: nn.Module
    train_data: DataLoader
    optimizer: torch.optim.Optimizer
    gpu_id: int = 0
    save_every: int = 10
    loss_fn: nn.Module = field(default_factory=lambda: nn.CrossEntropyLoss())


class Trainer:
    def __init__(self, params: TrainerParams) -> None:
        self.gpu_id = params.gpu_id
        self.model = params.model
        self.model = self.model.to(self.gpu_id)
        self.train_data = params.train_data
        self.optimizer = params.optimizer
        self.loss_fn = params.loss_fn
        self.save_every = params.save_every

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data):,}")
        for i, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = self.loss_fn(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            self.optimizer.step()
            print(f"Step {i+1}/{len(self.train_data)} | Loss: {loss.item():.3f}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
                
class TinyLLMDataset(TorchDataset):

    def __init__(self, shards_path, T, split, total_tokens):
        self.T = T
        self.total_tokens = total_tokens
        assert split in {'train', 'val'}, f"split {split} must be either 'train' or 'val'"
        shards = os.listdir(shards_path)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(shards_path, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])

    def __len__(self):
        return self.total_tokens//(self.T+1)

    def _load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def __getitem__(self, idx):
        while len(self.tokens) < (self.T + 1): # not enough tokens left in this shard
            self.current_shard = (self.current_shard + 1) % len(self.shards) # move to next shard
            self.tokens = torch.cat([self.tokens, self._load_tokens(self.shards[self.current_shard])], dim=0) # load next shard and append to remaining tokens
        buf = self.tokens[:self.T+1]
        X = buf[:-1]
        Y = buf[1:]
        self.tokens = self.tokens[self.T+1:] # move window
        return X, Y
    
