import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch import Tensor

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