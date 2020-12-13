"""
Tests comparing HuggingFace layers to our previous implementations
(mostly from the Annotated Transformer).

In each test we generally compare three implementations: our original,
the HuggingFace implementation, and f'{network_name}HF', which is
our code modified to reproduce HF results. We do this to ensure
that we are aware of the discrepancies between our previous
implementations and that there are no unknown discrepancies.

There is no test for positional encodings
"""
# pylint: disable=missing-function-docstring,invalid-name

import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from transformers import activations, configuration_bert, modeling_bert

import old_layers

# pylint: disable=invalid-name


def init_linear_seed(layer, seed):
    """
    Initializes manual seed for Linear layers
    """
    for module in layer.modules():
        if isinstance(module, torch.nn.Linear):
            torch.manual_seed(seed)
            module.reset_parameters()


class ExampleTest(unittest.TestCase):
    """
    Example test. For torch version>=1.4.0, transformers.activations.gelu is
    defined to be F.gelu, so this should always pass.
    """

    def test_new_gelu(self):
        x = (np.random.rand(5, 5) - 1) * 10.0
        x = torch.Tensor(x)
        self.assertTrue(torch.eq(activations.gelu(x), F.gelu(x)).all().item())

    def test_new_gelu_grad(self):
        x = (np.random.rand(1) - 1) * 10.0
        y = copy.deepcopy(x)
        x = torch.DoubleTensor(x)
        y = torch.DoubleTensor(y)
        x.requires_grad = True
        y.requires_grad = True
        out_x = activations.gelu(x)
        out_y = F.gelu(y)
        out_x.backward()
        out_y.backward()

        self.assertTrue(torch.eq(x.grad, y.grad).all().item())
        self.assertTrue(torch.autograd.gradcheck(activations.gelu, x))
        self.assertTrue(torch.autograd.gradcheck(F.gelu, y))


class GELUTest(unittest.TestCase):
    """
    Differences:
        Old GELU is the GELU used by OpenAI in GPT.
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
            * (x + 0.044715 * torch.pow(x, 3))))
        HF (and torch) uses the slightly different GELU originally used by
        Google in their official BERT Repo.
            x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        Google has since changed their GELU implementation to match the OpenAI
        implementation.
    """

    def test_new_gelu(self):
        x = (np.random.rand(5, 5) - 1) * 10.0
        x = torch.DoubleTensor(x)
        old_output = old_layers.GELU()(x)
        new_output = activations.gelu(x)
        hf_output = old_layers.GELUHF()(x)
        self.assertFalse(torch.allclose(new_output, old_output))
        self.assertTrue(torch.allclose(hf_output, new_output))


class AttentionTest(unittest.TestCase):
    """
    Differences:
        Last linear layer only in old attention.

        Old attention uses clones to clone single linear layer four
        times(q,k,v,last), this means they're all initialised the same,
        while new attention has three independently initialised linear
        layers.

    TODO: Test masking and dropout.
    """

    def test(self):
        """
        Success condition:
            Our original attention is different from the hf attention,
            while our new implementation agrees with the hf attention.
        """
        hf_bert_config = configuration_bert.BertConfig(
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        torch.manual_seed(0)
        old_att = old_layers.MultiHeadedAttention(h=2, d_model=8, dropout=0.0)
        torch.manual_seed(0)
        new_att = old_layers.MultiHeadedAttentionHF(h=2, d_model=8, dropout=0.0)
        torch.manual_seed(0)
        hf_att = modeling_bert.BertSelfAttention(hf_bert_config)

        x = (np.random.rand(4, 12, 8) - 1) * 10.0
        x = torch.Tensor(x)

        old_output = old_att(x, x, x, mask=None)
        new_output = new_att(x, x, x, mask=None)
        hf_output = hf_att(x)[0]

        self.assertFalse(torch.eq(old_output, hf_output).all().item())
        self.assertTrue(torch.eq(new_output, hf_output).all().item())


class LayerNormTest(unittest.TestCase):
    # pylint: disable=line-too-long
    r""" Test Layer Normalization.
    Diffs:
        The epsilon in ours is outside the square root. This is different
        from the description in the original paper
        (<https://arxiv.org/abs/1607.06450>).
        Ours:
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x]} + \epsilon} * \gamma + \beta
        Theirs:
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        We use the unbiased estimator \frac{(X_i - \mu)^2}{n-1} while
        the HuggingFace version (which is nn.LayerNorm) uses the biased
        \frac{(X_i - \mu)^2}{n}. The original paper uses the latter.
    """

    def test(self):
        """
        Success condition:
            Our original LN is different from the hf LN,
            while our new implementation agrees with the hf LN.
        """
        size = 8
        eps = 1e-5
        x = (np.random.rand(7, 15, size) - 1) * 10.0
        x = torch.DoubleTensor(x)

        torch.manual_seed(1234)
        old_ln = old_layers.LayerNorm(size, eps)
        torch.manual_seed(1234)
        # This is defined to be nn.LayerNorm as of transformers 2.11.0
        new_ln = old_layers.LayerNormHF(size, eps).double()
        torch.manual_seed(1234)
        hf_ln = modeling_bert.BertLayerNorm(size, eps).double()

        old_output = old_ln(x)
        new_output = new_ln(x)
        hf_output = hf_ln(x)

        self.assertFalse(torch.allclose(new_output, old_output))
        self.assertTrue(torch.allclose(new_output, hf_output))


class PositionalEmbeddingTest(unittest.TestCase):
    """
    AM-v2 uses sinusoidal (relative) positional embeddings, following
    the original Attention is All You Need paper, but virtually all
    future work uses absolute embeddings and sinusoidal embeddings
    are not implemented in HuggingFace.

    There are no tests for our positional embeddings for now, since
    our code is an exact copy of the positional encodings code in
    the official PyTorch tutorial at
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def test(self):
        pass


class FeedForwardTest(unittest.TestCase):
    """Feed Forward Network Unit Testing
    Diffs:
        Added Layer Normalization
        Our GeLU function is different from HF's GeLU
        HF uses pytorch.nn.functional's GeLU activation function
        Our GeLU function : check AMV 3-10
        HF adds the last linear layer to BertOutput when calculating
        residual connection
        Our model includes this layer at the end of the
        PositionwiseFeedForward Layer
        Our model's residual connection : check AMV 3-13
    """

    def test(self):
        """
        Success condition:
            Our original FFN is different from the hf FFN,
            while our new implementation agrees with the hf FFN.
        """
        hf_bert_config = configuration_bert.BertConfig(
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

        torch.manual_seed(0)
        new_ffn = old_layers.PositionwiseFeedForwardHF(d_model=8, d_ff=3072, dropout=0)

        x = (np.random.rand(4, 12, 8) - 1) * 10.0
        x = torch.Tensor(x)
        new_output = new_ffn(x)

        torch.manual_seed(0)
        hf_output = modeling_bert.BertIntermediate(hf_bert_config)(x)

        self.assertTrue(torch.eq(new_output, hf_output).all().item())


class SublayerConnectionTest(unittest.TestCase):
    """Residual Connection Test
    Diffs:
        Added a function init_linear_seed for manual seeding
        for modules with more than 2 layers (for old_layers)

    Description :
        In HuggingFace, the function applying residual connection
        includes a linear layer.
        Our model includes this linear layer at the end of Multi-
        HeadedAttention and PositionwiseFeedFoward layer
        To accurately verify that the new version of our model and
        HF is in accord, no_last_layer is implemented in both MHA and
        PFF. If no_last_layer is True, the last layer from each MHA and
        PFF are removed, and vice versa.

    """

    def test(self):
        hf_bert_config = configuration_bert.BertConfig(
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

        torch.manual_seed(0)
        new_sublayer_connection = old_layers.SublayerConnectionHF(size=8, dropout=0)

        x = torch.Tensor(np.random.rand(4, 12, 8) - 1) * 10.0

        ## Test residual connection for ffn and attention
        y_ffn = old_layers.PositionwiseFeedForwardHF(d_model=8, d_ff=3072, dropout=0)
        init_linear_seed(y_ffn, 0)

        y_att = old_layers.MultiHeadedAttentionHF(h=2, d_model=8, dropout=0.0)
        init_linear_seed(y_att, 0)

        new_output_ffn = new_sublayer_connection(
            x, lambda x: (y_ffn(x, no_last_layer=False))
        )
        new_output_att = new_sublayer_connection(
            x, lambda x: y_att(x, x, x, no_last_layer=False)
        )

        hf_ffn1 = modeling_bert.BertIntermediate(hf_bert_config)
        hf_ffn2 = modeling_bert.BertOutput(hf_bert_config)
        init_linear_seed(hf_ffn1, 0)
        init_linear_seed(hf_ffn2, 0)

        hf_output_ffn = hf_ffn2(hf_ffn1(x), x)

        hf_att1 = modeling_bert.BertSelfAttention(hf_bert_config)
        hf_att2 = modeling_bert.BertSelfOutput(hf_bert_config)
        init_linear_seed(hf_att1, 0)
        init_linear_seed(hf_att2, 0)

        hf_output_att = hf_att2(hf_att1(x)[0], x)
        self.assertTrue(torch.eq(new_output_ffn, hf_output_ffn).all().item())
        self.assertTrue(torch.eq(new_output_att, hf_output_att).all().item())


class EncoderTest(unittest.TestCase):
    """
    Diffs :
        EncoderLayerHF matches FeedForward Layer's output dimension
        which is 3072 in Bert
    """

    def test(self):
        hf_bert_config = configuration_bert.BertConfig(
            hidden_size=8,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

        x = torch.Tensor(np.random.rand(4, 12, 8) - 1) * 10.0

        new_enc = old_layers.EncoderLayerHF(size=8, h=2, dropout=0)
        init_linear_seed(new_enc, 0)
        new_output = new_enc(x, mask=None)

        hf_enc = modeling_bert.BertLayer(hf_bert_config)
        init_linear_seed(hf_enc, 0)
        hf_output = hf_enc(x)[0]

        self.assertTrue(torch.eq(new_output, hf_output).all().item())


if __name__ == "__main__":
    unittest.main()
