# set up 

conda create --name makemore python=3.11 
conda activate makemore 
pip install tensorboard torch biotite 


# to train on a FASTA dataset 

python makemore.py -i dataset/bglb.fa -o bglb 


# if you just want to sample from a model (data will be written in FASTA format) 

python makemore.py -o bglb --sample_only 



