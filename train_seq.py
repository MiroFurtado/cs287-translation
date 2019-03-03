# Naive Basil Pesto - Miro Furtado & Simon Shen
import torch
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
from tqdm import tqdm

def eval_perplexity(encoder, decoder, corpus_iter):
    """Evaluate the perplexity of a seq2seq model on a given corpus.
    """

    encoder.eval() #no dropout :(
    decoder.eval()

    loss_func = ntorch.nn.CrossEntropyLoss(reduction="none").spec("vocab")

    total_tokens = 0
    total_loss = 0

    for batch in tqdm(corpus_iter, position=0):
        _, hidden = encoder(batch.src)
        preds, _ = decoder(batch.trg, hidden)
        
        preds_n = preds[{"trgSeqlen": slice(0,preds.size("trgSeqlen")-1)}]
        trg_n = batch.trg[{"trgSeqlen": slice(1,preds.size("trgSeqlen"))}]
        
        loss = loss_func(preds_n, trg_n)
        loss = loss*(trg_n!=1).float() #only credit for non-pad predictions
        
        total_loss += loss.sum(('trgSeqlen','batch')).item() #sum up all the loss
        total_tokens += (trg_n!=1).long().sum(('trgSeqlen','batch')).item() #add all non-padding tokens
    ppl = np.exp((total_loss/total_tokens))
    return ppl

def train_seq2seq(encoder, decoder, corpus_data, num_epochs=10, lr=0.01, bsz=32):
    """Trains a basic seq2seq model.

    Parameters
    ----------
    encoder : ntorch.nn.Module
        The `LSTM` that will encode the input.
    decoder: ntorch.nn.Module
        Decodes input
    corpus_data : (train : torchtext.datasets, val : torchtext.datasets)
        The data that is going to be used for training and in validation
    """
    train_iter, val_iter = data.BucketIterator.splits(corpus_data, batch_size=bsz, device=device,
                                                    repeat=False, sort_key=lambda x: len(x.src))

    encoder.train() #
    decoder.train()

    loss_func = ntorch.nn.CrossEntropyLoss().spec("vocab")
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch in tqdm(train_iter, position=0):
            
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            
            _, hidden = encoder(batch.src)
            preds, _ = decoder(batch.trg, hidden)
            
            #Miro 2:30 PM 3/1/19: Shift prediction vs target - don't predict identity mapping
            preds_n = preds[{"trgSeqlen": slice(0,preds.size("trgSeqlen")-1)}]
            trg_n = batch.trg[{"trgSeqlen": slice(1,preds.size("trgSeqlen"))}]
            
            loss = loss_func(preds_n, trg_n)
            loss.backward()
            encoder_opt.step()
            decoder_opt.step()
    print("\nEPOCH %d: Loss %f, val perplexity %f"\
            % (epoch, loss.item(), eval_perplexity(encoder, decoder, val_iter)))
    encoder.train()
    decoder.train()

