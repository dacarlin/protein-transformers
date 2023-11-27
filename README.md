# Simple, hackable protein transformers 

Inspired by Karpathy's makemore (which deploys several different models for a character-level next-token prediction task), the goal of this project is to provide a simple, hackable implementation of protein transformer models for education and research. 


## Install 

Install with Conda 

```
conda create --name protein python=3.11 
conda activate protein 
pip install tensorboard torch biotite 
```

## Training example

As an example, we can use sequence homologs from the [BglB family](https://github.com/dacarlin/enzyme-ml-benchmarks) to train a protein transformer capable of designing new enzymes that fold and function the same way as the proteins in the training set 

To train on a FASTA dataset to replicate Experiment 1  

```
python train.py -i dataset/dhfr/dataset.fa -o experiments/experiment_1 
```

Once the model is trained, and you would like to sample (data will be written in FASTA format):

```
python train.py -o experiments/experiment_1 --sample-only 
```

This will create new protein sequences that could potentially fold and function
like the proteins in the training set. 


## Model 

The model used here is a transformer based on GPT-2, trained with a next-token completion task.
[This wonderful article illustrates all the aspects of the GPT-2 model](https://jalammar.github.io/illustrated-gpt2/). 
GPT models are decoder-only transformers, in contrast to BERT, which is an encoder-only model. 
GPT was chosen here since it's a simple and approchable start for using transformer models
for protein design 


