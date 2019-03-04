import torch
from namedtensor import ntorch, NamedTensor

class DecoderAttn(ntorch.nn.Module):
    """Decoder based on the implementation in Yuntian's slides.
    """

    def __init__(self, hidden_dim = 512, num_layers = 2, dropout = 0.5, trg_vocab_len = 11560, attn_length=20):
        super(DecoderAttn, self).__init__()

        self.embedding = ntorch.nn.Embedding(trg_vocab_len, hidden_dim).spec("trgSeqlen", "embedding")

        self.attn_transform = ntorch.nn.Linear(2*hidden_dim, attn_length).spec("hidden", "attn")

        self.dropout = ntorch.nn.Dropout(dropout)
        self.LSTM = ntorch.nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout).spec("embedding", "trgSeqlen", "hidden")
        self.out = ntorch.nn.Linear(hidden_dim, trg_vocab_len).spec("hidden", "vocab")

    def forward(self, input, hidden, decoder_context):
        """Forward pass
        
        Parameters (all NamedTensors)
        ----------
        input : (trgSeqlen, batch)
        hidden : tuple((batch, layers, hidden) , ((batch, layers, hidden)))
        decoder_context : (batch, srcSeqlen, hidden)

        Todo
        ----
        * Where do we apply dropout?
        * Read over the Slack for description

        """

        x = self.embedding(input)
        x = self.dropout(x)

        x, hidden = self.LSTM(x, hidden)

        # (batch, srcSeqlen, hidden) x (batch, trgSeqlen, hidden) -> (batch, trgSeqlen, srcSeqlen)
        attn_weights = x.dot("hidden", decoder_context).softmax("srcSeqlen")
        context = attn_weights.dot('srcSeqlen', decoder_context) # (batch, trgSeqlen, hidden)

        x = self.dropout(x)
        y = self.out(x)
        # No softmax because cross-entropy
        return y, hidden