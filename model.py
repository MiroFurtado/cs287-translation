import torch
from namedtensor import ntorch, NamedTensor

class S2SNet(ntorch.nn.Module):
  def __init__(self, hidden_dim=512, num_layers=2, dropout=0.5):
    super(S2SNet, self).__init__()
    self.encoder = EncoderS2S(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    self.decoder = DecoderS2S(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

  def forward(self, input):
    context, hidden = self.encoder(input)
    preds, _ = self.decoder(input, hidden, context)
    return preds[{"trgSeqlen": slice(0,preds.size("trgSeqlen")-1)}]

class AttnNet(ntorch.nn.Module):
  def __init__(self, hidden_dim=512, num_layers=2, dropout=0.5, n=2):
    super(AttnNet, self).__init__()
    self.encoder = EncoderS2S(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    self.decoder = DecoderAttn(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, n=n)

  def forward(self, input):
    context, hidden = self.encoder(input)
    preds, _ = self.decoder(input, hidden, context)
    return preds[{"trgSeqlen": slice(0,preds.size("trgSeqlen")-1)}]

class EncoderS2S(ntorch.nn.Module):
  def __init__(self, hidden_dim = 512, num_layers = 2, dropout=0.5, src_vocab_len = 13353):
    super(EncoderS2S, self).__init__()
    
    self.embedding = ntorch.nn.Embedding(src_vocab_len, hidden_dim).spec("srcSeqlen", "embedding")
    self.dropout = ntorch.nn.Dropout(dropout)
    self.LSTM = ntorch.nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout).spec("embedding", "srcSeqlen", "hidden")
  
  def forward(self, input, hidden=None): 
    # NOTE: `input` must have its named dimensions correctly ordered: ("srcSeqlen", "batch")
    input = ntorch.tensor(torch.flip(input.values, (0,)),("srcSeqlen","batch")) #reverse input, improves translation
    x = self.embedding(input) 
    x = self.dropout(x)
    x, hidden = self.LSTM(x)
    x = self.dropout(x) #Miro 3/7/19 12:00 PM - Add dropout
    return x, hidden

class DecoderS2S(ntorch.nn.Module):
  def __init__(self, hidden_dim = 512, num_layers = 2, dropout = 0.5, trg_vocab_len = 11560):
    super(DecoderS2S, self).__init__()
    
    self.embedding = ntorch.nn.Embedding(trg_vocab_len, hidden_dim).spec("trgSeqlen", "embedding")
    self.dropout = ntorch.nn.Dropout(dropout)
    self.LSTM = ntorch.nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout).spec("embedding", "trgSeqlen", "hidden")
    self.out = ntorch.nn.Linear(hidden_dim, trg_vocab_len).spec("hidden", "vocab")
    
  def forward(self, input, hidden, unk=None):
    x = self.embedding(input)
    x = self.dropout(x)
    x, hidden = self.LSTM(x, hidden)
    x = self.dropout(x)
    y = self.out(x)
    # No softmax because cross-entropy
    return y, hidden

class DecoderAttn(ntorch.nn.Module):
    """Decoder based on the implementation in Yuntian's slides and Luong.
    """

    def __init__(self, hidden_dim = 512, num_layers = 2, dropout = 0.5, trg_vocab_len = 11560, n=2):
        super(DecoderAttn, self).__init__()

        self.embedding = ntorch.nn.Embedding(trg_vocab_len, hidden_dim).spec("trgSeqlen", "embedding")


        self.dropout = ntorch.nn.Dropout(dropout)
        self.LSTM = ntorch.nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout).spec("embedding", "trgSeqlen", "hidden")
        self.h2h = ntorch.nn.Linear(2*hidden_dim, n*hidden_dim).spec("hidden", "hidden") #reduce to hidden
        self.out = ntorch.nn.Linear(n*hidden_dim, trg_vocab_len).spec("hidden", "vocab")

    def get_context(self, hidden, decoder_context):
        """(batch, srcSeqlen, hidden) x (batch, trgSeqlen, hidden) -> (batch, trgSeqlen, srcSeqlen)
        """

        attn_weights = hidden.dot("hidden", decoder_context).softmax("srcSeqlen")
        context = attn_weights.dot("srcSeqlen", decoder_context) # (batch, trgSeqlen, hidden)
        return context

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
        x = self.dropout(x) #Miro 4:15 PM 3/4/19 - Changing dropout location
        
        context = self.get_context(x, decoder_context)
        x = self.h2h(ntorch.cat([x,context], "hidden")).relu()
        y = self.out(x)
        # No softmax because cross-entropy
        return y, hidden