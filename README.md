# Simple, hackable protein transformers 

The goal of this project is to provide a simple, hackable implementation of protein transformer models for education and research. Inspired by Karpathy's simple, hackable approach in the [makemore](https://github.com/karpathy/makemore) series. 


## Code contents 

There are three main sections, all implemented in PyTorch:

- protein data loaders and tokenizers 
- models (our first model is a "decoder only" transformer)
- training loop 


### Protein data loaders 

There are two `Dataset`-like classes defined: `ProteinDataset` uses a character-level tokenizer (as used by ESM and all other protein language models I know about) and `BpeDataset`, which uses a BytePair encoding for tokenization. 

Find the `ProteinDataset` and `BpeDataset` classes defined in `data.py`. An example of using the `ProteinDataset` class: 

```python 
train_dataset = ProteinDataset(train_proteins, chars, max_word_length)
```

### Transformer model 

The model implemented here is a standard "decoder-only" transformer architecture
("decoder-only" meaning that we use a triangular mask and ask the model to predict
the next token). Following makemore, the implementation is totally spelled out in 
Python code so we can see all the details. 


### Training loop 

[To do]


## Compute environment 

To run the code locally on Metal, simply pass `--device mps` to the main script. 
To install the dependencies (just PyTorch and Biotite, optionally Hugging Face
tokenizers if needed)

```
# virtual environment 
python -m venv .venv 
python -m pip install torch biotite tokenizers tensorboard 
```

## Training from scratch  

To instantiate a new model and train on a FASTA dataset, use the following command line. 
The example dataset in the repo contains 26,878 homologs of HypF that are between 64 and
128 residues in length and contain predicted catalytic residue sites from UniProt. 

To train a model on this dataset, use the following command: 

```
python main.py -i hypf.da -o hypf 
```

Once the model is trained, and you would like to sample (data will be written in FASTA format):

```
python main.py -o hypf --sample-only 
```





