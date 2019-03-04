# Naive Basil Pesto - Miro Furtado & Simon Shen
import torch
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
from tqdm import tqdm
import numpy as np
import argparse
import spacy
import model_seq


def eval_perplexity(encoder, decoder, corpus_iter):
    """Evaluate the perplexity of a seq2seq model on a given corpus.
    """

    encoder.eval() #no dropout :(
    decoder.eval()

    loss_func = ntorch.nn.CrossEntropyLoss(reduction="none").spec("vocab")

    total_tokens = 0
    total_loss = 0

    for batch in tqdm(corpus_iter):
        context, hidden = encoder(batch.src)
        preds, _ = decoder(batch.trg, hidden, context)
        
        preds_n = preds[{"trgSeqlen": slice(0,preds.size("trgSeqlen")-1)}] # XXXXXXX_
        trg_n = batch.trg[{"trgSeqlen": slice(1,preds.size("trgSeqlen"))}] # _XXXXXXX
        
        loss = loss_func(preds_n, trg_n)
        loss = loss*(trg_n!=1).float() #only credit for non-pad predictions
        
        total_loss += loss.sum(('trgSeqlen','batch')).item() #sum up all the loss
        total_tokens += (trg_n!=1).long().sum(('trgSeqlen','batch')).item() #add all non-padding tokens
    ppl = np.exp((total_loss/total_tokens))
    return ppl

def train_model(encoder, decoder, corpus_data, num_epochs=10, lr=0.001, bsz=32, prefix = "checkpoint"):
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
    
    assert(torch.cuda.is_available())
    device = torch.device('cuda:0')
    train_iter, val_iter = data.BucketIterator.splits(corpus_data, batch_size=bsz, device=device,
                                                    repeat=False, sort_key=lambda x: len(x.src))

    encoder.train() #
    decoder.train()

    loss_func = ntorch.nn.CrossEntropyLoss().spec("vocab")
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    ppl = 10000

    for epoch in range(num_epochs):
        for batch in tqdm(train_iter):
            
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            
            decoder_context, hidden = encoder(batch.src)
            preds, _ = decoder(batch.trg, hidden, decoder_context)
            
            #Miro 2:30 PM 3/1/19: Shift prediction vs target - don't predict identity mapping
            preds_n = preds[{"trgSeqlen": slice(0,preds.size("trgSeqlen")-1)}]
            trg_n = batch.trg[{"trgSeqlen": slice(1,preds.size("trgSeqlen"))}]
            
            loss = loss_func(preds_n, trg_n)
            loss.backward() #backprop thru loss
            encoder_opt.step()
            decoder_opt.step() #descend!
        temp_ppl =  eval_perplexity(encoder, decoder, val_iter)
        print("\n[***] EPOCH %d: Loss %f, val perplexity %f"\
                % (epoch, loss.item(), temp_ppl)) #update
        if epoch % 5 == 0:
            if temp_ppl < ppl:
                print("[***] Saving model with improved perplexity")
                ppl = temp_ppl
                torch.save(decoder.state_dict(), prefix+"_e"+epoch+"_"+"decoder"+".w")
                torch.save(encoder.state_dict(), prefix+"_p"+ppl+"_e"+epoch+"_"+"encoder"+".w")

        encoder.train() #turn dropout back on
        decoder.train()

def parse_arguments():
    "Parse arguments from console"
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('--epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('--prefix', type=string, default="checkpoint",
                   help='Prefix for model checkpointing')
    p.add_argument('--lr', type=float, default=0.0001,
                   help='learning rate for adam')
    p.add_argument('--bsz', type=int, default=32,
                   help='batch size for train')
    return p.parse_args()
    
def generate_data():
    "Generate data from spacy: de -> en"
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = NamedField(names=('srcSeqlen',), tokenize=tokenize_de)
    EN = NamedField(names=('trgSeqlen',), tokenize=tokenize_en,
                    init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS
    MAX_LEN = 20
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 5
    DE.build_vocab(train.src, min_freq=MIN_FREQ)
    EN.build_vocab(train.trg, min_freq=MIN_FREQ)

    return train, val, test

def main():
    "Entrance function for running from console"
    args = parse_arguments()
    print("[*] Preparing data: 🇩🇪  -> 🇬🇧")
    train, val, _ = generate_data() #throw away test just to be safe!

    print("[*] Building initial model on CUDA")
    encoder = model_seq.EncoderS2S().cuda()
    decoder = model_seq.DecoderS2S().cuda()
    print("\t🧗 Begin loss function descent")
    train_model(encoder, decoder, (train, val), num_epochs=args.epochs, lr=args.lr, bsz=args.bsz, prefix=args.prefix)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)