# Simple, hackable protein transformers 

Inspired by Karpathy's makemore, the goal of this project is to provide a simple hackable implementation of protein transformer models for education and research. 


## Install 

Install with Conda 

```
conda create --name ppt python=3.11 
conda activate ppt 
pip install tensorboard torch biotite 
```

## Example: training on a homologous sequence family 

As an example, we can use sequence homologs from the [BglB family](https://github.com/dacarlin/enzyme-ml-benchmarks) to train a protein transformer capable of designing new enzymes that fold and function the same way as the proteins in the training set 

To train on a FASTA dataset: 

```
python ppt.py -i dataset/bglb.fa -o bglb 
```

Once the model is trained, and you would like to sample (data will be written in FASTA format):

```
python ppt.py -o bglb --sample-only 
```



