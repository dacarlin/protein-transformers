# Experiment 1 

Train a small transformer on a small dataset of homologs to test if the code is configured correctly 

Dataset: 

N homologs of BglB 

15,222 proteins between 256 and 512 in length, homologs of the BglB protein in the benchmark datasets, which is approximately 6 million tokens. 

Let's start with our model size should be about 1/20 of that in terms of number of parameters, so 292,262 parameters! Let's start with that!  



- Model: 8 layers, model size 128, trained for X steps 
- Training: batches of 128 for N 




# Experiment 2 

Byte pair encoding on the same dataset as above! 


# Experiment 3

DHFR 


# [Planned] Experiment 4

Train an ensemble of tall skinny models (meaning, small embedding size and lots of laters). Ensemble is train 8 of them each with a specialty... how to define this specialty, and how to combine with a MOE style? 


