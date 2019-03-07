# Naive Basil Pesto - Miro Furtado & Simon Shen
import torch
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
from tqdm import tqdm
import numpy as np
import argparse
import spacy
from model import *


def eval_perplexity(model, corpus_iter):
    """Evaluate the perplexity of a seq2seq model on a given corpus.
    """

    model.eval()

    loss_func = ntorch.nn.CrossEntropyLoss(reduction="none", ignore_index=1).spec("vocab")

    total_tokens = 0
    total_loss = 0

    for batch in tqdm(corpus_iter):
        with torch.no_grad():
            preds = model(batch.src)
            
            #Miro 2:30 PM 3/1/19: Shift prediction vs target - don't predict identity mapping
            trg_n = batch.trg[{"trgSeqlen": slice(1,preds.size("trgSeqlen"))}]
            
            loss = loss_func(preds, trg_n)
            #loss = loss*(trg_n!=1).float() #only credit for non-pad predictions
            
            total_loss += loss.sum(('trgSeqlen','batch')).item() #sum up all the loss
            total_tokens += (trg_n!=1).long().sum(('trgSeqlen','batch')).item() #add all non-padding tokens
    ppl = np.exp((total_loss/total_tokens))
    return ppl

def train_model(model, corpus_data, num_epochs=10, lr=0.001, bsz=32, prefix = "checkpoint", weight_decay=0.):
    """Trains a basic seq2seq model.

    Parameters
    ----------
    model : ntorch.nn.Module
    corpus_data : (train : torchtext.datasets, val : torchtext.datasets)
        The data that is going to be used for training and in validation
    """
    
    assert(torch.cuda.is_available())
    print("\t\t Hyperparameters: lr %f, epochs %d, bsz %d, decay %f" %(lr, num_epochs, bsz, weight_decay))
    device = torch.device('cuda:0')
    train_iter, val_iter = data.BucketIterator.splits(corpus_data, batch_size=bsz, device=device,
                                                    repeat=False, sort_key=lambda x: len(x.src))

    loss_func = ntorch.nn.CrossEntropyLoss(ignore_index=1).spec("vocab")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    ppl = eval_perplexity(model, val_iter)
    print("[***] Starting ppl %f" % ppl)

    model.train()

    for epoch in range(num_epochs):
        for batch in tqdm(train_iter):
            
            opt.zero_grad()
            
            preds = model(batch.src)
            
            #Miro 2:30 PM 3/1/19: Shift prediction vs target - don't predict identity mapping
            trg_n = batch.trg[{"trgSeqlen": slice(1,preds.size("trgSeqlen"))}]
            
            loss = loss_func(preds, trg_n)
            loss.backward() #backprop thru loss
            opt.step()
        temp_ppl =  eval_perplexity(model, val_iter)
        print("\n[***] EPOCH %d: Loss %f, val perplexity %f"\
                % (epoch, loss.item(), temp_ppl)) #update
        if epoch % 5 == 0:
            if temp_ppl < ppl:
                print("[***] Saving model with improved perplexity")
                ppl = temp_ppl
                torch.save(model.state_dict(), prefix+"_p"+str(ppl)+"_e"+str(epoch)+"_"+"model"+".w")

        model.train() #turn dropout back on

def parse_arguments():
    "Parse arguments from console"
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('--epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('--prefix', default="checkpoint",
                   help='Prefix for model checkpointing')
    p.add_argument('--model_path', default=None,
                   help='Path to model')
    p.add_argument('--lr', type=float, default=0.0001,
                   help='learning rate for adam')
    p.add_argument('--decay', type=float, default=0.0005,
                   help='weight decay')
    p.add_argument('--bsz', type=int, default=32,
                   help='batch size for train')
    p.add_argument('--attn', action='store_true')
    p.add_argument('--n', type=int, default=2,
                   help='hidden state multiplier')
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
    if args.attn:
        print("[*] Starting train of attention model")
    else:
        print("[*] Starting train of seq2seq model")
    print("[*] Preparing data: 🇩🇪  -> 🇬🇧")
    train, val, _ = generate_data() #throw away test just to be safe!

    print("[*] Building initial model on CUDA")
    if args.attn:
        model = AttnNet(n=args.n).cuda()
    else:
        model = S2SNet().cuda()
    if args.model_path:
        print("[*] Loading model from file")
        model.load_state_dict(torch.load(args.model_path))
    print("\t🧗 Begin loss function descent")
    train_model(model, (train, val), num_epochs=args.epochs, lr=args.lr,\
         bsz=args.bsz, prefix=args.prefix, weight_decay=args.decay)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)