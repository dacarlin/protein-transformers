"""Adapted from Karpathy's makemore. Thanks Andrej!"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from biotite.sequence.io.fasta import FastaFile 


# -----------------------------------------------------------------------------
# Configuration class 
@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1] 
    n_layer: int = 4
    embed_dim: int = 64
    n_head: int = 4

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        # output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.embed_dim = config.embed_dim

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.embed_dim, 4 * config.embed_dim),
            c_proj  = nn.Linear(4 * config.embed_dim, config.embed_dim),
            act     = nn.GELU(approximate="tanh"), 
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_dim),
            wpe = nn.Embedding(config.block_size, config.embed_dim),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.embed_dim),
        ))
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, embed_dim)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, embed_dim)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, greedy=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either take most likely (greedy) or sample from the distribution 
        if greedy:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, greedy=False).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        
        # better than this, calculate percent ID to each member of train set,
        # but this entails alignment and expensive calculation, so just detect 
        # 100% identity here 
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    
    print(f"Printing {num} samples from the model:")
    groups = [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]
    for lst, desc in groups:
        print(f"{len(lst)} samples that are {desc}:")
        for i, word in enumerate(lst):
            header = f"sample_{i + 1} {desc}"
            print(f">{header}\n{word}")
    print("Done printing samples")


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

# ---------------------------------------------------------------------
# Datasets for protein sequences and BytePair-encoded protein sequences 

class ProteinDataset(Dataset):
    """Dataset for protein sequences with character-level tokenization"""

    def __init__(self, proteins, chars, max_protein_length):
        self.proteins = proteins
        self.chars = chars
        self.max_protein_length = max_protein_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.proteins)

    def contains(self, word):
        return word in self.proteins

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_protein_length + 1 # <START> token followed by proteins

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.proteins[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_protein_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_protein_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y





def create_datasets(input_file):

    # preprocessing of the input text file
    proteins = []
    fasta_file = FastaFile.read(input_file) 
    for header, sequence in fasta_file.items():
        proteins.append(sequence)
    max_word_length = max(len(w) for w in proteins)

    # partition the input data into a training and the test set
    test_set_size = int(len(proteins) * 0.1) # 10% of the training set
    rp = torch.randperm(len(proteins)).tolist()
    train_proteins = [proteins[i] for i in rp[:-test_set_size]]
    test_proteins = [proteins[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_proteins)} training examples and {len(test_proteins)} test examples")

    chars = sorted(list(set(''.join(proteins)))) # all the possible characters
    tokens = sum(len(w) for w in proteins)
    
    print("using characters as tokens")
    print(f"number of examples in the dataset: {len(proteins)}")
    print(f"max protein length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))
    print(f"total tokens: {tokens}")

    # wrap in dataset objects
    train_dataset = ProteinDataset(train_proteins, chars, max_word_length)
    test_dataset = ProteinDataset(test_proteins, chars, max_word_length)

    return train_dataset, test_dataset


# -----------------------------------------------------------------------------
# Parse arguments and run the training loop 
if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="Make more proteins")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='hypf.fa', help="input fasta file")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--max-steps', type=int, default=75_000, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=42, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads")
    parser.add_argument('--embed-dim', type=int, default=64, help="number of feature channels in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=4e-3, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       embed_dim=args.embed_dim)
    model = Transformer(config)
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt'), map_location=torch.device(args.device)))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay, 
        betas=(0.9, 0.99), eps=1e-8)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        pin_memory=True)

    # training loop
    best_loss = None
    step = 0
    for batch in train_loader:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            print_samples(num=10)  # do this a final time at the end 
            break

