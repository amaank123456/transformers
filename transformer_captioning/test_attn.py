import unittest
import torch
import torch.nn.functional as F
from transformer import AttentionLayer  # Adjust import based on file structure

class TestAttentionLayer(unittest.TestCase):
    
    def setUp(self):
        # Set up the AttentionLayer instance and test parameters
        embed_dim = 64
        dropout = 0.1
        self.attention_layer = AttentionLayer(embed_dim, dropout)
        self.query = torch.randn(2, 10, embed_dim)  # Batch size 2, sequence length 10
        self.key = torch.randn(2, 15, embed_dim)    # Batch size 2, sequence length 15
        self.value = torch.randn(2, 15, embed_dim)  # Batch size 2, sequence length 15
        self.attn_mask = torch.ones(2, 10, 15)  # Compatibility with query/key-value dimensions

    def test_forward_output_shape(self):
        # Test if output shape matches expectations
        output = self.attention_layer(self.query, self.key, self.value, self.attn_mask)
        self.assertEqual(output.shape, (2, 10, self.attention_layer.embed_dim))
    
    def test_forward_output_non_nan(self):
        # Test if output does not contain NaN values
        output = self.attention_layer(self.query, self.key, self.value, self.attn_mask)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")

    def test_dropout_effect(self):
        # Test dropout effect (check if dropout reduces values when enabled)
        self.attention_layer.dropout.p = 1.0  # Set dropout to maximum to zero-out values
        output = self.attention_layer(self.query, self.key, self.value)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)), "Output should be zero with full dropout")

    def test_no_dropout(self):
        # Test with dropout disabled to confirm that forward pass is stable
        self.attention_layer.dropout.p = 0.0
        output1 = self.attention_layer(self.query, self.key, self.value)
        output2 = self.attention_layer(self.query, self.key, self.value)
        self.assertTrue(torch.allclose(output1, output2), "Output varies with dropout disabled")

if __name__ == "__main__":
    unittest.main()
