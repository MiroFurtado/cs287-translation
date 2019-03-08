# Naive Basil Pesto - Miro Furtado & Simon Shen
# Acknowledgement: Thanks to CS287 TFs https://colab.research.google.com/drive/1kPE8v6j9aRO1xxRRxM-YLXI_5TVwLZlg.
import torch, argparse, model, daft, pickle
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
from tqdm import tqdm
from collections import namedtuple
import matplotlib.pyplot as plt

# HELPERS
def unsqueeze(tens, dim):
    existing_dim = tens.dims[0]
    return tens._split(existing_dim, (existing_dim, dim), {dim: 1})
def escape_bleu(l):
    l = l.split()
    l.append('</s>')
    return ' '.join(l[:l.index('</s>')])+"\n"

def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")
# HELPERS - performs operation on each element of tuple
def get_each(tup, dim, idx):
    return tuple(part[{dim: idx}] for part in tup)
def clone_each(tup):
    return tuple(part.clone() for part in tup)
def unsqueeze_each(tup, dim):
    return tuple(unsqueeze(part, dim) for part in tup)

# DISPLAY BEAM
def display_beam(stack, EN_vocab, show_token=False):
    # If show_token then each node's content is the index of the word. Otherwise, it's the actual word
    def get_node_name(t, word):
        return EN_vocab.itos[word] + str(t)
    maxlen = len(stack)
    num_hypotheses_out = stack[-1].beam_words.shape["beam"]
    pgm = daft.PGM([maxlen + 2, num_hypotheses_out + 1], origin=[-2, -1], grid_unit=4, node_unit=2.5)
    # Beginning of sentence node
    pgm.add_node(daft.Node(
        name    = "<s>-1",
        content = "<s>",
        x       = -1,
        y       = 0
    ))
    for t, beam in enumerate(stack):
        edges = {}
        for idx, word in enumerate(beam.beam_words.squeeze("trgSeqlen").tolist()):
            this_node_name = get_node_name(t, word)

            # Add unique nodes to graph
            if word not in edges:
              edges[word] = {}
              pgm.add_node(daft.Node(
                  name    = this_node_name, 
                  content = (EN_vocab.itos[word] if show_token else str(word)) + "\n" + '({:.3f})'.format(beam.beam_scores[{"beam": idx}].item()),
                  x       = t,
                  y       = idx
              ))

            # Keep track of unique edges and num times they occur
            prev_word = 2 # beginning of sentence
            if t > 0:
              prev_word = stack[t - 1].beam_words[{"beam": beam.beam_prev_idxs[{"beam": idx}].item()}].item()
            if prev_word not in edges[word]:
              edges[word][prev_word] = 0
            edges[word][prev_word] += 1

        # Add unique edges to graph, weighted by num times occur
        for word, neighbors in edges.items():
            this_node_name = get_node_name(t, word)
            for prev_word, edge_weight in neighbors.items():
                prev_node_name = get_node_name(t - 1, prev_word)
                pgm.add_edge(
                    prev_node_name, 
                    this_node_name, 
                    linewidth = edge_weight,
                    edgecolor = (0, 0, 0, 0.4),
                    facecolor = (0, 0, 0, 0.4)
                )

    pgm.render()

# Actual beam search happens here in beam_decode
Beam = namedtuple(
    "Beam",
    [ "beam_words"
    , "beam_prev_idxs"
    , "beam_scores"
    , "beam_state"
    ],
)

# EOS on beam
EOS = namedtuple(
    "EOS",
    [ "beam_t" # when
    , "beam_idx" # which position on the beam
    ],
)

Hypothesis = namedtuple(
    "Hypothesis",
    [ "words"
    , "scores"
    , "avgscore"
    ],
)

def get_top_k(K, scores, state, disallowed_preds=[]):
    V = scores.shape["vocab"]

    # Set score of disallowed_preds to -inf
    for v in disallowed_preds:
        scores[{"vocab": v}] = - float("Inf")

    # Stack beam and vocab -- consider all hypotheses at this point
    scores = scores.stack(("beam", "vocab"), "beam")

    # Keep the most probably K hypotheses
    kscores, kidxs = scores.topk("beam", K, largest=True, sorted=True)

    # Interpret the best hypothesis i.e. what is the new word and what was the previous hypothesis
    kprev_idxs = kidxs.div(V)
    kwords = kidxs.fmod(V)

    # And keep the corresponding hidden/cell states for the best hypotheses
    kstates = get_each(state, "beam", kprev_idxs.squeeze("trgSeqlen"))

    return kscores, kwords, kprev_idxs, kstates

def run_decoder(decoder_model, x, hidden, batch_dim, encoded_context=None):
    # Forward computes decoder_model, but allows the batch dimension to be called something else e.g. beam
    # Also calculates logsoftmax at the end (not included in decoder model)

    x = x.rename(batch_dim, "batch")
    hidden = tuple(part.rename(batch_dim, "batch") for part in hidden)

    if encoded_context is None:
        x, hidden = decoder_model(x, hidden)
    else:
        x, hidden = decoder_model(x, hidden, encoded_context)
    x = x.log_softmax("vocab")

    x = x.rename("batch", batch_dim)
    hidden = tuple(part.rename("batch", batch_dim) for part in hidden)

    return x, hidden

def check_eos(kwords, kscores, early_eos, t):
    # If EOS appears at top of beam, returns True. If eos appears otherwise, then records appearance in early_eos, sets score of eos to -Inf, and returns False.
    if (kwords == 3).any(): # if eos is on beam
        eos_idx = (kwords == 3).argmax().item() # get index of eos
        if eos_idx == 0: # if eos appears at the top of the beam
            return True
        else:
            kscores[{"beam": eos_idx}] = - float('Inf')
            early_eos.append(EOS(
                beam_t = t,
                beam_idx = eos_idx
            ))
    return False

def backtrack(beam_idx, stack, device, maxlen = None):
  # Will add padding to get to maxlen if not None
  words  = []
  scores = []

  for beam in reversed(stack): # start from most recent word -> first word
      words.append(beam.beam_words[{"beam": beam_idx}])
      scores.append(beam.beam_scores[{"beam": beam_idx}])
      beam_idx = beam.beam_prev_idxs[{"beam": beam_idx}].item()

  avgscore = scores[0].item() / len(words) # equation 63 of Neubig, G. Neural Machine Translation
  
  padding_ones = ntorch.ones(maxlen - len(words), names=("trgSeqlen",)).to(device)
  
  return Hypothesis(
      words  = ntorch.cat(list(reversed(words)) + [1 * padding_ones.clone().to(torch.long)], "trgSeqlen"),
      scores = ntorch.cat(list(reversed(scores)) + [scores[0] * padding_ones.clone()], "trgSeqlen"),
      avgscore = avgscore
  )
  
def beam_decode(encoded_summary, decoder_model, maxlen, beam_width, device, encoded_context=None, num_hypotheses_out=None):
    # for Kaggle submissions, use maxlen=3 and beam_width=10
    # at the last trgSeqlen, don't pick the top `beam_width`, pick the top `num_hypotheses_out` hypotheses
    if num_hypotheses_out is None:
        num_hypotheses_out = beam_width

    early_eos = []
    stack = []

    # First step
    kwords = ntorch.tensor([[2]], names=("beam", "trgSeqlen")).to(device) # <s>
    kstates = unsqueeze_each(encoded_summary, "beam")
    kscores = ntorch.zeros((beam_width, 1), names=("beam", "trgSeqlen")).to(device)

    for t in range(maxlen):
        # Calculate probability of next word given history
        scores, state = run_decoder(decoder_model, kwords, kstates, encoded_context=encoded_context, batch_dim="beam")

        # Cumulative log prob
        tscores = scores
        K = beam_width
        if t > 0:
            tscores += kscores
            # When quitting, choose the top num_hypotheses_out, not the top beam_width hypotheses
            if t == maxlen - 1:
                K = num_hypotheses_out

        # Choose top K by cumulative log probability
        kscores, kwords, kprev_idxs, kstates = get_top_k(K, tscores, state, disallowed_preds=[0, 1, 2]) # disallow unk, pad, <s>
        bm = Beam(
            beam_words     = kwords.clone(),
            beam_prev_idxs = kprev_idxs.clone(),
            beam_scores    = kscores.clone(),
            beam_state     = clone_each(kstates)
        )
        
        # Check for eos
        eos_result = check_eos(kwords, kscores, early_eos, t)
        if eos_result:
            # When quitting early, choose the top num_hypotheses_out, not the top beam_width hypotheses
            kscores, kwords, kprev_idxs, kstates = get_top_k(num_hypotheses_out, tscores, state, disallowed_preds=[0, 1, 2]) # disallow unk, pad, <s>
            stack.append(Beam(
                beam_words     = kwords.clone(),
                beam_prev_idxs = kprev_idxs.clone(),
                beam_scores    = kscores.clone(),
                beam_state     = clone_each(kstates)
            ))
            break
        
        # Add beam to stack
        stack.append(bm)
            
    hypotheses = []
    # Consider on-beam hypotheses
    for hypothesis_idx in range(num_hypotheses_out):
        hypotheses.append(backtrack(hypothesis_idx, stack, device, maxlen))
    # Consider hypotheses with early eos
    for eos in early_eos:
        hypotheses.append(backtrack(eos.beam_idx, stack[:(eos.beam_t + 1)], device, maxlen))

    # Choose the top num_hypotheses_out by average log probabilities
    hypotheses.sort(key=lambda hyp: hyp.avgscore, reverse=True)
    return (
        ntorch.cat([unsqueeze(x.words, "beam") for x in hypotheses[:num_hypotheses_out]], "beam"), 
        ntorch.cat([unsqueeze(x.scores, "beam") for x in hypotheses[:num_hypotheses_out]], "beam"), 
        ntorch.cat([ntorch.tensor([x.avgscore], names=("beam")) for x in hypotheses[:num_hypotheses_out]], "beam"),
        stack
    )

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
    p.add_argument('--trg', type=open,
                   help='English text file of true translations')
    p.add_argument('-k', '--beam_width', type=int, default=10,
                   help='Beam width')
    p.add_argument('--hypotheses', type=int, default=100,
                   help='Number of hypotheses to be outputted')
    p.add_argument('--maxlen', type=int, default=3,
                   help='Maximum hypothesis length')
    p.add_argument('--linenum', type=int, default=None,
                   help='Line in source file to be translated')
    p.add_argument('--attn', action='store_true')
    p.add_argument('--bleu', action='store_true',
                   help='Write predictions to file in BLEU calculation format')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--prefix', default="beamsearch",
                   help='Prefix for filenames')
    p.add_argument('--writepreds', action='store_true',
                   help='Write predictions to file in Kaggle submission format')
    p.add_argument('--printpreds', action='store_true',
                   help='Print predictions and average log probabilities')
    p.add_argument('--writebeam', action='store_true',
                   help='Saves beam graph')
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

    if args.writepreds:
        f = open(args.prefix + "_preds.txt", "w")
        f.write("Id,Predicted\n")
    if args.bleu:
        f_bleu = open(args.prefix + "_bleu_preds.txt", "w")
    for i, sentence in tqdm(enumerate(args.src), position=0):
        if args.trg is not None:
            en_sentence = args.trg.readline()
        if args.linenum is not None and not args.linenum == i:
            continue
        de_sentence = [DE_vocab.stoi[word] for word in sentence.split(" ")]
        de_sentence = ntorch.tensor([de_sentence], names=("batch", "srcSeqlen")).to(device)
        de_sentence = de_sentence.transpose("srcSeqlen", "batch")
        encoded_context, encoded_summary = mdl.encoder(de_sentence)
        encoded_summary = get_each(encoded_summary, "batch", 0) #squeeze batch dim
        encoded_context = encoded_context[{"batch": 0}]

        words, _, avgscores, stack = beam_decode(encoded_summary, mdl.decoder, maxlen=args.maxlen, beam_width=args.beam_width, device=device, encoded_context=encoded_context, num_hypotheses_out=args.hypotheses)

        if args.writepreds:
            f.write(str(i) + "," + ' '.join(['|'.join([escape(EN_vocab.itos[i]) for i in words[{"beam": h}].tolist()]) for h in range(args.hypotheses)]) + "\n")
        if args.bleu:
            sentence = ' '.join([EN_vocab.itos[i] for i in words[{"beam": 0}].tolist()]) # the hypotheses are already sorted, so the top hypothesis is the best one
            f_bleu.write(escape_bleu(sentence))
        if args.printpreds:
            tqdm.write("\n  GERMAN: " + ' '.join([DE_vocab.itos[i] for i in de_sentence.squeeze("batch").tolist()]))
            if args.trg is not None:
                tqdm.write(" ENGLISH: " + en_sentence)
            for h in range(args.hypotheses):
                tqdm.write('{:.5f}: '.format(avgscores[{"beam": h}].item()) + ' '.join([EN_vocab.itos[i] for i in words[{"beam": h}].tolist()]))
        if args.writebeam:
            display_beam(stack, EN_vocab, show_token=True)
            plt.gca().invert_yaxis()
            plt.savefig(args.prefix + "_beam_%03d.png" % i)
    if args.writepreds:
        f.close()
    if args.bleu:
        f_bleu.close()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)