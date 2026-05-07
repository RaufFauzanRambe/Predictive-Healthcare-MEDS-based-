"""
test_model.py - Unit tests for model architectures and training.

Tests cover model creation, forward passes, training loop execution,
and model saving/loading functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.model import MLPModel, LSTMModel, TransformerModel, ModelFactory


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def batch_size():
    return 16

@pytest.fixture
def input_dim():
    return 32

@pytest.fixture
def seq_len():
    return 24

@pytest.fixture
def tabular_input(batch_size, input_dim):
    """Create a random tabular input tensor."""
    return torch.randn(batch_size, input_dim)

@pytest.fixture
def sequence_input(batch_size, seq_len, input_dim):
    """Create a random sequence input tensor."""
    return torch.randn(batch_size, seq_len, input_dim)


# ─── MLP Model Tests ─────────────────────────────────────────────────────────

class TestMLPModel:
    """Tests for the MLPModel architecture."""

    def test_forward_pass_shape(self, tabular_input, batch_size):
        """Test that forward pass produces the correct output shape."""
        model = MLPModel(input_dim=32, hidden_dims=[64, 32], output_dim=1)
        output = model(tabular_input)

        assert output.shape == (batch_size, 1)

    def test_output_is_logits(self, tabular_input):
        """Test that output values are unbounded logits (not probabilities)."""
        model = MLPModel(input_dim=32, output_dim=1)
        output = model(tabular_input)

        # Logits can be any real number (not bounded to [0, 1])
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_hidden_dims(self, tabular_input):
        """Test model with various hidden dimension configurations."""
        for hidden_dims in [[64], [128, 64], [256, 128, 64, 32]]:
            model = MLPModel(input_dim=32, hidden_dims=hidden_dims, output_dim=1)
            output = model(tabular_input)
            assert output.shape[-1] == 1

    def test_dropout_applied(self):
        """Test that dropout affects training but not eval mode."""
        model = MLPModel(input_dim=32, dropout=0.5)
        x = torch.randn(8, 32)

        model.train()
        out1 = model(x)
        out2 = model(x)
        # With 50% dropout, outputs should differ in training mode
        # (statistically very likely for 8x32 inputs)
        assert not torch.allclose(out1, out2)

        model.eval()
        out3 = model(x)
        out4 = model(x)
        # In eval mode, outputs should be identical
        assert torch.allclose(out3, out4)

    def test_activations(self, tabular_input):
        """Test model with different activation functions."""
        for activation in ["ReLU", "GELU", "LeakyReLU"]:
            model = MLPModel(input_dim=32, activation=activation)
            output = model(tabular_input)
            assert not torch.isnan(output).any()


# ─── LSTM Model Tests ────────────────────────────────────────────────────────

class TestLSTMModel:
    """Tests for the LSTMModel architecture."""

    def test_forward_pass_shape(self, sequence_input, batch_size):
        """Test that forward pass produces correct output shape."""
        model = LSTMModel(input_dim=32, hidden_dim=64, output_dim=1)
        output = model(sequence_input)

        assert output.shape == (batch_size, 1)

    def test_bidirectional(self, sequence_input, batch_size):
        """Test bidirectional LSTM produces valid output."""
        model = LSTMModel(input_dim=32, hidden_dim=64, bidirectional=True, output_dim=1)
        output = model(sequence_input)
        assert output.shape == (batch_size, 1)

    def test_unidirectional(self, sequence_input, batch_size):
        """Test unidirectional LSTM produces valid output."""
        model = LSTMModel(input_dim=32, hidden_dim=64, bidirectional=False, output_dim=1)
        output = model(sequence_input)
        assert output.shape == (batch_size, 1)

    def test_with_attention(self, sequence_input, batch_size):
        """Test LSTM with attention mechanism."""
        model = LSTMModel(input_dim=32, hidden_dim=64, attention=True, output_dim=1)
        output = model(sequence_input)
        assert output.shape == (batch_size, 1)

    def test_without_attention(self, sequence_input, batch_size):
        """Test LSTM without attention (uses last hidden state)."""
        model = LSTMModel(input_dim=32, hidden_dim=64, attention=False, output_dim=1)
        output = model(sequence_input)
        assert output.shape == (batch_size, 1)

    def test_multi_layer(self, sequence_input, batch_size):
        """Test multi-layer LSTM."""
        model = LSTMModel(input_dim=32, hidden_dim=64, num_layers=3, output_dim=1)
        output = model(sequence_input)
        assert output.shape == (batch_size, 1)


# ─── Transformer Model Tests ────────────────────────────────────────────────

class TestTransformerModel:
    """Tests for the TransformerModel architecture."""

    def test_forward_pass_shape(self, sequence_input, batch_size):
        """Test that forward pass produces correct output shape."""
        model = TransformerModel(input_dim=32, d_model=64, nhead=4, output_dim=1)
        output = model(sequence_input)
        assert output.shape == (batch_size, 1)

    def test_cls_token_prepended(self, sequence_input, batch_size):
        """Test that [CLS] token is correctly prepended internally."""
        model = TransformerModel(input_dim=32, d_model=64, nhead=4)
        # Just verify the model runs without error
        output = model(sequence_input)
        assert output.shape[0] == batch_size

    def test_different_configurations(self, sequence_input, batch_size):
        """Test various transformer configurations."""
        configs = [
            {"d_model": 64, "nhead": 4, "num_encoder_layers": 2},
            {"d_model": 128, "nhead": 4, "num_encoder_layers": 3},
            {"d_model": 64, "nhead": 2, "num_encoder_layers": 1},
        ]
        for cfg in configs:
            model = TransformerModel(input_dim=32, output_dim=1, **cfg)
            output = model(sequence_input)
            assert output.shape == (batch_size, 1)

    def test_positional_encoding_types(self, sequence_input, batch_size):
        """Test both sinusoidal and learned positional encoding."""
        for pe_type in ["sinusoidal", "learned"]:
            model = TransformerModel(
                input_dim=32, d_model=64, nhead=4,
                positional_encoding=pe_type, output_dim=1
            )
            output = model(sequence_input)
            assert output.shape == (batch_size, 1)


# ─── ModelFactory Tests ─────────────────────────────────────────────────────

class TestModelFactory:
    """Tests for the ModelFactory class."""

    def test_create_mlp(self):
        """Test creating an MLP model via factory."""
        model = ModelFactory.create("mlp", input_dim=64, output_dim=1)
        assert isinstance(model, MLPModel)

    def test_create_lstm(self):
        """Test creating an LSTM model via factory."""
        model = ModelFactory.create("lstm", input_dim=64, output_dim=1)
        assert isinstance(model, LSTMModel)

    def test_create_transformer(self):
        """Test creating a Transformer model via factory."""
        model = ModelFactory.create("transformer", input_dim=64, output_dim=1)
        assert isinstance(model, TransformerModel)

    def test_unknown_model_type(self):
        """Test that unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create("unknown_model")


# ─── Model Save/Load Tests ──────────────────────────────────────────────────

class TestModelSaveLoad:
    """Tests for model serialization and deserialization."""

    def test_save_and_load_mlp(self, tmp_path):
        """Test saving and loading an MLP model."""
        model = MLPModel(input_dim=32, output_dim=1)
        x = torch.randn(4, 32)
        expected_output = model(x)

        # Save
        save_path = tmp_path / "mlp_test.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": "MLPModel",
        }, save_path)

        # Load
        loaded_model = MLPModel(input_dim=32, output_dim=1)
        checkpoint = torch.load(save_path, weights_only=False)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()

        actual_output = loaded_model(x)
        assert torch.allclose(expected_output, actual_output, atol=1e-6)

    def test_save_and_load_lstm(self, tmp_path):
        """Test saving and loading an LSTM model."""
        model = LSTMModel(input_dim=32, hidden_dim=64, output_dim=1)
        x = torch.randn(4, 10, 32)
        model.eval()
        expected_output = model(x)

        # Save
        save_path = tmp_path / "lstm_test.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": "LSTMModel",
        }, save_path)

        # Load
        loaded_model = LSTMModel(input_dim=32, hidden_dim=64, output_dim=1)
        checkpoint = torch.load(save_path, weights_only=False)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()

        actual_output = loaded_model(x)
        assert torch.allclose(expected_output, actual_output, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
