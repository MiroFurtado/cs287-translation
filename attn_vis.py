# Naive Basil Pesto - Miro Furtado & Simon Shen
import torch, argparse, model, daft, pickle
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
from tqdm import tqdm
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

# HELPERS - performs operation on each element of tuple
def get_each(tup, dim, idx):
    return tuple(part[{dim: idx}] for part in tup)
def unsqueeze(tens, dim):
    existing_dim = tens.dims[0]
    return tens._split(existing_dim, (existing_dim, dim), {dim: 1})
def unsqueeze_each(tup, dim):
    return tuple(unsqueeze(part, dim) for part in tup)

def parse_arguments():
    "Parse arguments from console"
    def open_rb(file):
        return open(file, "rb")
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('vocab', type=open_rb,
                   help='File of DE vocab and EN vocab as NamedFields. Suggested: "DE_EN_vocab.pkl"')
    p.add_argument('model', type=open_rb,
                   help='File of weights for wrapped model')
    p.add_argument('src', type=open,
                   help='German text file to be translated')
    p.add_argument('--maxlen', type=int, default=3,
                   help='Maximum hypothesis length')
    p.add_argument('--linenum', type=int, default=0,
                   help='Zero-indexed line number of sentence to be translated')
    p.add_argument('--attn', action='store_true')
    p.add_argument('--cuda', action='store_true')
    return p.parse_args()

def main():
    "Entrance function for running from console"
    args = parse_arguments()
    device = "cuda" if args.cuda else "cpu"
	
    print("[*] Loading vocab: DE -> EN")
    DE_vocab = pickle.load(args.vocab)
    EN_vocab = pickle.load(args.vocab)

    print("[*] Loading models on %s" % device)
    mdl_weights = torch.load(args.model, map_location=device)
    if args.attn:
        mdl = model.AttnNet().to(device)
    else:
        mdl = model.S2SNet().to(device)
    mdl.load_state_dict(mdl_weights)
    mdl.eval()
    
    print("[*] Translating")
    for i, sentence in tqdm(enumerate(args.src), position=0):
        if i < args.linenum:
            continue
        de_sentence = [DE_vocab.stoi[word] for word in sentence.split(" ")]
        de_sentence = ntorch.tensor([de_sentence], names=("batch", "srcSeqlen")).to(device)
        de_sentence = de_sentence.transpose("srcSeqlen", "batch")
        encoded_context, encoded_summary = mdl.encoder(de_sentence)
        encoded_context = encoded_context[{"batch": 0}]

        # First step
        word = ntorch.tensor([[2]], names=("batch", "trgSeqlen")).to(device) # <s>
        state = encoded_summary

        preds = []
        attn = []

        for t in range(args.maxlen):
            # Calculate probability of next word given history
            scores, state, attn_weights = mdl.decoder(word, state, encoded_context, return_attn=True)
            word = scores.argmax("vocab")
            preds.append(word.detach())
            attn.append(attn_weights.detach())
            if word.item() == 3:
                break
        preds = ntorch.cat(preds, "trgSeqlen").squeeze("batch")
        attn = ntorch.cat(attn, "trgSeqlen").squeeze("batch")
        plt.imshow(np.flip(attn.numpy(), 1), cmap="gray")
        plt.colorbar()
        plt.xlabel("German")
        plt.ylabel("English")
        plt.xticks(np.arange(de_sentence.shape["srcSeqlen"]), [DE_vocab.itos[i] for i in de_sentence[{"batch": 0}].tolist()], rotation=45)
        plt.yticks(np.arange(preds.shape["trgSeqlen"]), [EN_vocab.itos[i] for i in preds.tolist()])
        plt.show()
        break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)