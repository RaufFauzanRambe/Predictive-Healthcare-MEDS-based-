"""
model.py - Neural network architectures for predictive healthcare modeling.

Implements three model architectures:
1. MLP: Multi-layer perceptron for tabular feature prediction
2. LSTM: Bidirectional LSTM with attention for time-series prediction
3. Transformer: Encoder-only transformer for sequence modeling

All models output logits for binary classification (e.g., 30-day readmission).
Use BCEWithLogitsLoss during training for numerical stability.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# MLP Model
# =============================================================================

class MLPModel(nn.Module):
    """
    Multi-layer perceptron for tabular healthcare data.

    A straightforward feed-forward network with configurable hidden dimensions,
    dropout, batch normalization, and activation. Suitable for static features
    (demographics, vitals, lab values at a single time point).

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer sizes.
        output_dim: Number of output logits (1 for binary classification).
        dropout: Dropout probability between layers.
        activation: Activation function name ('ReLU', 'GELU', 'LeakyReLU').
        use_batch_norm: Whether to use batch normalization.
        use_layer_norm: Whether to use layer normalization.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: list[int] = None,
        output_dim: int = 1,
        dropout: float = 0.3,
        activation: str = "ReLU",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Select activation function
        act_fn = self._get_activation(activation)

        # Build layers
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation — logits for BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Return the activation module by name."""
        activations = {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "LeakyReLU": nn.LeakyReLU(0.01),
        }
        return activations.get(name, nn.ReLU())

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization for ReLU-like networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output logits of shape (batch_size, output_dim).
        """
        return self.network(x)


# =============================================================================
# LSTM Model with Attention
# =============================================================================

class LSTMAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism for LSTM outputs.

    Computes attention weights over the sequence of LSTM hidden states
    and returns a weighted context vector. This allows the model to
    focus on the most relevant time steps for the prediction.

    Args:
        hidden_dim: Dimension of LSTM hidden states.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.

        Args:
            lstm_outputs: LSTM output of shape (batch_size, seq_len, hidden_dim).

        Returns:
            Tuple of (context_vector, attention_weights):
            - context_vector: (batch_size, hidden_dim)
            - attention_weights: (batch_size, seq_len)
        """
        # Compute attention scores
        scores = self.attention(lstm_outputs).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum of LSTM outputs
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)  # (batch, hidden)
        return context, weights


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM with optional attention for healthcare time-series.

    Processes patient event sequences and predicts binary outcomes.
    The bidirectional structure captures both past and future context
    within the lookback window, and the attention mechanism highlights
    the most informative time steps.

    Args:
        input_dim: Feature dimension per time step.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        output_dim: Number of output logits.
        dropout: Dropout probability.
        bidirectional: Whether to use bidirectional LSTM.
        attention: Whether to apply attention on LSTM outputs.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = attention
        self.num_directions = 2 if bidirectional else 1

        # Input projection (optional dimensionality adjustment)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        lstm_output_dim = hidden_dim * self.num_directions
        if attention:
            self.attention_layer = LSTMAttention(lstm_output_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LSTM and linear layers."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim).

        Returns:
            Output logits of shape (batch_size, output_dim).
        """
        # Project input if needed
        x = self.input_proj(x)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden * num_directions)

        # Apply attention or use last hidden state
        if self.use_attention:
            context, _ = self.attention_layer(lstm_out)
        else:
            # Use the last time step output
            context = lstm_out[:, -1, :]

        # Classification
        logits = self.classifier(context)
        return logits


# =============================================================================
# Transformer Model
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal or learned positional encoding for transformer inputs.

    Injects positional information into the input embeddings so the
    model can distinguish the order of events in a patient timeline.

    Args:
        d_model: Model dimension.
        max_seq_len: Maximum sequence length.
        learned: If True, use learned positional embeddings instead of sinusoidal.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 512,
        learned: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learned = learned

        if learned:
            self.pe = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        else:
            # Sinusoidal positional encoding
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Positional-encoded tensor of the same shape.
        """
        if self.learned:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Encoder-only Transformer for healthcare time-series prediction.

    Uses multi-head self-attention to capture complex temporal dependencies
    across patient events. The [CLS] token approach aggregates sequence
    information into a single representation for classification.

    Args:
        input_dim: Feature dimension per time step.
        d_model: Internal model dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of transformer encoder layers.
        dim_feedforward: Feed-forward dimension in encoder blocks.
        output_dim: Number of output logits.
        dropout: Dropout probability.
        activation: Activation in feed-forward ('relu' or 'gelu').
        max_seq_len: Maximum input sequence length.
        positional_encoding: 'sinusoidal' or 'learned'.
    """

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 512,
        positional_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        learned = positional_encoding == "learned"
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            learned=learned,
            dropout=dropout,
        )

        # Learnable [CLS] token for sequence classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Prepends a [CLS] token, adds positional encoding, passes through
        the transformer encoder, and uses the [CLS] output for classification.

        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim).

        Returns:
            Output logits of shape (batch_size, output_dim).
        """
        batch_size = x.size(0)

        # Project input to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len+1, d_model)

        # Extract [CLS] token output
        cls_output = self.layer_norm(x[:, 0, :])  # (batch, d_model)

        # Classification
        logits = self.classifier(cls_output)
        return logits


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """
    Factory for creating model instances from configuration.

    Provides a unified interface to instantiate any supported model type
    with the appropriate architecture parameters from model_config.yaml.
    """

    _registry: dict[str, type[nn.Module]] = {
        "mlp": MLPModel,
        "lstm": LSTMModel,
        "transformer": TransformerModel,
    }

    @classmethod
    def create(cls, model_type: str, **kwargs) -> nn.Module:
        """
        Create a model instance by type name.

        Args:
            model_type: One of 'mlp', 'lstm', 'transformer'.
            **kwargs: Architecture-specific keyword arguments.

        Returns:
            Instantiated PyTorch model.

        Raises:
            ValueError: If the model type is not registered.
        """
        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(cls._registry.keys())}"
            )

        model_class = cls._registry[model_type]
        model = model_class(**kwargs)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Created {model_type} model with {param_count:,} parameters")
        return model

    @classmethod
    def from_config(cls, model_config: dict) -> nn.Module:
        """
        Create a model from a full model_config dictionary.

        Reads the 'active_model' key to determine which architecture
        to instantiate, then passes the corresponding sub-config.

        Args:
            model_config: Full model_config.yaml contents.

        Returns:
            Instantiated PyTorch model.
        """
        model_type = model_config.get("active_model", "mlp")
        model_params = model_config.get(model_type, {})
        return cls.create(model_type, **model_params)
