import unittest
import torch
import math
from transformer import MultiHeadAttentionLayer  # Adjust import based on file structure

class TestMultiHeadAttentionLayer(unittest.TestCase):

    def setUp(self):
        embed_dim = 64
        num_heads = 4
        dropout = 0.1
        self.multi_head_attention = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)
        self.query = torch.randn(2, 10, embed_dim)  # Batch size 2, sequence length 10
        self.key = torch.randn(2, 15, embed_dim)    # Batch size 2, sequence length 15
        self.value = torch.randn(2, 15, embed_dim)  # Batch size 2, sequence length 15
        self.attn_mask = torch.ones(2, 1, 10, 15)  # Compatibility with query/key-value dimensions

    def test_forward_output_shape(self):
        # Test if output shape matches expectations
        output = self.multi_head_attention(self.query, self.key, self.value, self.attn_mask)
        self.assertEqual(output.shape, (2, 10, self.multi_head_attention.embed_dim))

    def test_forward_output_non_nan(self):
        # Test if output does not contain NaN values
        output = self.multi_head_attention(self.query, self.key, self.value, self.attn_mask)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")

    def test_no_dropout(self):
        # Test with dropout disabled to confirm that forward pass is stable
        self.multi_head_attention.dropout.p = 0.0
        output1 = self.multi_head_attention(self.query, self.key, self.value)
        output2 = self.multi_head_attention(self.query, self.key, self.value)
        self.assertTrue(torch.allclose(output1, output2), "Output varies with dropout disabled")

    def test_num_heads_split(self):
        # Test if embeddings are split correctly across heads in the forward pass
        H = self.multi_head_attention.num_heads
        N, S, D = self.query.shape
        # Project query, key, value, and check their reshaping
        query_proj = self.multi_head_attention.query_proj(self.query).view(N, S, H, D // H).transpose(1, 2)
        self.assertEqual(query_proj.shape, (N, H, S, D // H), "Query projection shape does not match expected shape after split")

if __name__ == "__main__":
    unittest.main()
