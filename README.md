# Predicting protein transmembrane topology from 3D structure

This project investigated whether you can predict protein topology from 3D structure instead of just the primary protein sequence. The dataset was provided by the current SOTA model DeepTMHMM and can be downloaded here https://dtu.biolib.com/DeepTMHMM and contains 3576 proteins. To get the proteins as 3D structures [AlphaFold DB](https://alphafold.ebi.ac.uk) was used. 3544 proteins of the original dataset was found in the AlphaFold DB and was the size of the dataset used for this work.

The 3D structures (.pdb files) were used as input for for a pre-trained graph neural nerwork, [ESM-IF1](https://github.com/facebookresearch/esm#invf) described as <code>GVPTransformer</code> in [Learning inverse folding from millions of predicted structures. (Hsu et al. 2022)](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2). This was done to obtain feature vectors capturing the geometric information of the input data. To obtain feature vectors for all proteins of the dataset instructions described in section [Encoder output as structure representation](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding) was followed. The necessary functions needed to obtain the feature vectors is shown in the cell below.  

```
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
```

## Data
The data needed to run this program is stored in 5 files/folders

  1) A csv file containing all protein ID's, their target values, lengths and information about which proteins belong to each fold for cross validation. 
  2) A numpy file just containing the protein ID's 
  3) A file containing the predictions of the SOTA model DeepTMHMM 
  4) A folder containing the 5 trained models from the 5-fold cross validation 
  5) A folder containing 3544 protein encodings as .pt file (NB, 3.35 GB) 

The first 3 mentioned files can be found under the folder 'data' in this repository, and the folder containing the models is called 'models'. 

The last folder containing the 3544 protein encodings can be downloaded from https://www.dropbox.com/scl/fo/fldja9rwecmcbgutn13nm/h?rlkey=m1mvj346eq7pgezr5fpxb7818&dl=0 

## Installation
Download this repository 
```
git clone https://github.com/Katja-Jagd/Protein_topology_prediction
```
## Train model
```
usage: python main.py [--epochs N] [--lr LR] [-b N] [--mode {train}]
```
### Optional arguments

```
--epochs N                     Number of total epochs to run
--lr LR, --learning-rate LR    Learning rate
-b N, --batch-size N           Batch size
--mode {train,test}            Only train developed so far
```
### Example
```
python main.py --epochs 100 --lr 0.0001 -b 30 --mode train
```
## Output files
Five models from the 5-fold cross-validation setup
- model.1.pt
- model.2.pt
- model.3.pt
- model.4.pt
- model.5.pt

## Training curves
This program integrated the tool [Weights and Biases](https://wandb.ai/site) to track the training progress (loss and accuracy). An account is needed. 

## Use models to predict topology 
Follow instructions in the [scripts/Test.ipynb](https://github.com/Katja-Jagd/Protein_topology_prediction/blob/main/scripts/Test.ipynb) notebook 

## Project workflow
<img src="https://github.com/Katja-Jagd/Protein_topology_prediction/blob/main/work_flow.png" width="500" height="500">
