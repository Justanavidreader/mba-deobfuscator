"""
Tests for model architecture components.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from src.constants import (
    NODE_DIM, HIDDEN_DIM, D_MODEL, VOCAB_SIZE, NUM_ENCODER_LAYERS,
    NUM_ENCODER_HEADS, NUM_DECODER_LAYERS, NUM_DECODER_HEADS, D_FF,
    FINGERPRINT_DIM, MAX_SEQ_LEN, MAX_OUTPUT_LENGTH, MAX_OUTPUT_DEPTH,
    NUM_EDGE_TYPES, GGNN_TIMESTEPS
)
from src.models.positional import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from src.models.heads import TokenHead, ComplexityHead, ValueHead
from src.models.encoder import (
    GATJKNetEncoder, GGNNEncoder, GraphReadout, FingerprintEncoder
)
from src.models.decoder import TransformerDecoderLayer, TransformerDecoderWithCopy
from src.models.full_model import MBADeobfuscator


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def sample_graph_batch(batch_size):
    """Create sample graph batch for testing."""
    graphs = []
    for _ in range(batch_size):
        num_nodes = torch.randint(5, 15, (1,)).item()
        x = torch.randn(num_nodes, NODE_DIM)

        edge_list = []
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        edge_type = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))

        graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        graphs.append(graph)

    return Batch.from_data_list(graphs)


@pytest.fixture
def sample_fingerprint(batch_size):
    """Create sample fingerprint for testing."""
    return torch.randn(batch_size, FINGERPRINT_DIM)


class TestPositionalEncoding:
    """Test positional encoding modules."""

    def test_sinusoidal_encoding_shape(self, batch_size, seq_len):
        """Test sinusoidal encoding output shape."""
        pe = SinusoidalPositionalEncoding()
        x = torch.randn(batch_size, seq_len, D_MODEL)
        out = pe(x)

        assert out.shape == (batch_size, seq_len, D_MODEL)

    def test_learned_encoding_shape(self, batch_size, seq_len):
        """Test learned encoding output shape."""
        pe = LearnedPositionalEncoding()
        x = torch.randn(batch_size, seq_len, D_MODEL)
        out = pe(x)

        assert out.shape == (batch_size, seq_len, D_MODEL)

    def test_sinusoidal_encoding_variable_length(self, batch_size):
        """Test sinusoidal encoding with different sequence lengths."""
        pe = SinusoidalPositionalEncoding()

        for seq_len in [8, 16, 32]:
            x = torch.randn(batch_size, seq_len, D_MODEL)
            out = pe(x)
            assert out.shape == (batch_size, seq_len, D_MODEL)


class TestHeads:
    """Test output head modules."""

    def test_token_head_shape(self, batch_size, seq_len):
        """Test token head output shape."""
        head = TokenHead()
        x = torch.randn(batch_size, seq_len, D_MODEL)
        out = head(x)

        assert out.shape == (batch_size, seq_len, VOCAB_SIZE)

    def test_complexity_head_shape(self, batch_size):
        """Test complexity head output shapes."""
        head = ComplexityHead()
        x = torch.randn(batch_size, D_MODEL)
        length_logits, depth_logits = head(x)

        assert length_logits.shape == (batch_size, MAX_OUTPUT_LENGTH)
        assert depth_logits.shape == (batch_size, MAX_OUTPUT_DEPTH)

    def test_value_head_shape(self, batch_size):
        """Test value head output shape and range."""
        head = ValueHead()
        x = torch.randn(batch_size, D_MODEL)
        value = head(x)

        assert value.shape == (batch_size, 1)
        assert torch.all(value >= 0) and torch.all(value <= 1)


class TestEncoder:
    """Test encoder modules."""

    def test_gat_encoder_shape(self, sample_graph_batch):
        """Test GAT encoder output shape."""
        encoder = GATJKNetEncoder()
        out = encoder(
            sample_graph_batch.x,
            sample_graph_batch.edge_index,
            sample_graph_batch.batch
        )

        assert out.shape == (sample_graph_batch.x.size(0), HIDDEN_DIM)

    def test_ggnn_encoder_shape(self, sample_graph_batch):
        """Test GGNN encoder output shape."""
        encoder = GGNNEncoder()
        out = encoder(
            sample_graph_batch.x,
            sample_graph_batch.edge_index,
            sample_graph_batch.edge_type,
            sample_graph_batch.batch
        )

        assert out.shape == (sample_graph_batch.x.size(0), HIDDEN_DIM)

    def test_graph_readout_shape(self, sample_graph_batch):
        """Test graph readout output shape."""
        encoder = GATJKNetEncoder()
        node_embeddings = encoder(
            sample_graph_batch.x,
            sample_graph_batch.edge_index,
            sample_graph_batch.batch
        )

        readout = GraphReadout()
        graph_embedding = readout(node_embeddings, sample_graph_batch.batch)

        batch_size = sample_graph_batch.batch.max().item() + 1
        assert graph_embedding.shape == (batch_size, HIDDEN_DIM)

    def test_fingerprint_encoder_shape(self, sample_fingerprint):
        """Test fingerprint encoder output shape."""
        encoder = FingerprintEncoder()
        out = encoder(sample_fingerprint)

        assert out.shape == (sample_fingerprint.size(0), HIDDEN_DIM)


class TestDecoder:
    """Test decoder modules."""

    def test_decoder_layer_shape(self, batch_size, seq_len):
        """Test decoder layer output shape."""
        layer = TransformerDecoderLayer(D_MODEL, NUM_DECODER_HEADS, D_FF, 0.1)

        x = torch.randn(batch_size, seq_len, D_MODEL)
        memory = torch.randn(batch_size, seq_len, D_MODEL)

        out, attn = layer(x, memory)

        assert out.shape == (batch_size, seq_len, D_MODEL)
        assert attn.shape == (batch_size, seq_len, seq_len)

    def test_decoder_with_copy_shape(self, batch_size, seq_len):
        """Test decoder with copy mechanism output shapes."""
        decoder = TransformerDecoderWithCopy()

        tgt = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        memory = torch.randn(batch_size, seq_len, D_MODEL)

        vocab_logits, copy_attn, p_gen = decoder(tgt, memory)

        assert vocab_logits.shape == (batch_size, seq_len, D_MODEL)
        assert copy_attn.shape == (batch_size, seq_len, seq_len)
        assert p_gen.shape == (batch_size, seq_len, 1)
        assert torch.all(p_gen >= 0) and torch.all(p_gen <= 1)

    def test_causal_mask(self):
        """Test causal mask correctness."""
        decoder = TransformerDecoderWithCopy()
        mask = decoder.generate_square_subsequent_mask(5)

        assert mask.shape == (5, 5)

        for i in range(5):
            for j in range(5):
                if j > i:
                    assert mask[i, j] == float('-inf')
                else:
                    assert mask[i, j] == 0


class TestFullModel:
    """Test full model integration."""

    def test_encoder_gat(self, sample_graph_batch, sample_fingerprint):
        """Test encoder with GAT."""
        model = MBADeobfuscator(encoder_type='gat')
        memory = model.encode(sample_graph_batch, sample_fingerprint)

        batch_size = sample_fingerprint.size(0)
        assert memory.shape == (batch_size, 1, D_MODEL)

    def test_encoder_ggnn(self, sample_graph_batch, sample_fingerprint):
        """Test encoder with GGNN."""
        model = MBADeobfuscator(encoder_type='ggnn')
        memory = model.encode(sample_graph_batch, sample_fingerprint)

        batch_size = sample_fingerprint.size(0)
        assert memory.shape == (batch_size, 1, D_MODEL)

    def test_decoder_step(self, sample_graph_batch, sample_fingerprint, batch_size, seq_len):
        """Test decoder step."""
        model = MBADeobfuscator(encoder_type='gat')
        memory = model.encode(sample_graph_batch, sample_fingerprint)

        tgt = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        decode_output = model.decode(tgt, memory)

        assert 'vocab_logits' in decode_output
        assert 'copy_attn' in decode_output
        assert 'p_gen' in decode_output

        assert decode_output['vocab_logits'].shape == (batch_size, seq_len, VOCAB_SIZE)
        assert decode_output['p_gen'].shape == (batch_size, seq_len, 1)

    def test_full_forward_pass(self, sample_graph_batch, sample_fingerprint, batch_size, seq_len):
        """Test full forward pass."""
        model = MBADeobfuscator(encoder_type='gat')

        tgt = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        output = model(sample_graph_batch, sample_fingerprint, tgt)

        assert 'vocab_logits' in output
        assert 'copy_attn' in output
        assert 'p_gen' in output
        assert 'length_pred' in output
        assert 'depth_pred' in output
        assert 'value' in output

        assert output['vocab_logits'].shape == (batch_size, seq_len, VOCAB_SIZE)
        assert output['length_pred'].shape == (batch_size, MAX_OUTPUT_LENGTH)
        assert output['depth_pred'].shape == (batch_size, MAX_OUTPUT_DEPTH)
        assert output['value'].shape == (batch_size, 1)

    def test_get_value(self, sample_graph_batch, sample_fingerprint, batch_size):
        """Test value head access."""
        model = MBADeobfuscator(encoder_type='gat')
        value = model.get_value(sample_graph_batch, sample_fingerprint)

        assert value.shape == (batch_size, 1)
        assert torch.all(value >= 0) and torch.all(value <= 1)

    def test_dimension_consistency(self, sample_graph_batch, sample_fingerprint):
        """Test dimension consistency throughout pipeline."""
        model = MBADeobfuscator(encoder_type='gat')

        batch_size = sample_fingerprint.size(0)
        seq_len = 10
        tgt = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

        memory = model.encode(sample_graph_batch, sample_fingerprint)
        assert memory.size(-1) == D_MODEL

        decode_out = model.decode(tgt, memory)
        assert decode_out['vocab_logits'].size(-1) == VOCAB_SIZE

        output = model(sample_graph_batch, sample_fingerprint, tgt)
        assert output['vocab_logits'].size(1) == seq_len
        assert output['vocab_logits'].size(2) == VOCAB_SIZE


class TestParameterRegistration:
    """Test that parameters are properly registered."""

    def test_all_parameters_registered(self):
        """Test that all model parameters are registered."""
        model = MBADeobfuscator(encoder_type='gat')

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

        named_params = dict(model.named_parameters())
        assert 'graph_encoder.node_embedding.weight' in named_params
        assert 'decoder.token_embedding.weight' in named_params
        assert 'token_head.proj.weight' in named_params
        assert 'complexity_head.length_head.weight' in named_params
        assert 'value_head.mlp.0.weight' in named_params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
