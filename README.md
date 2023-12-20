# Predicting protein transmembrane topology from 3D structure

This project investigated whether you can predict protein topology from 3D structure instead of just the primary protein sequence. The dataset was provided by the current SOTA model DeepTMHMM and can be downloaded here https://dtu.biolib.com/DeepTMHMM and contains 3576 proteins. To get the proteins as 3D structures [AlphaFold DB](https://alphafold.ebi.ac.uk) was used. 3544 proteins of the original dataset was found in the AlphaFold DB and was the size of the dataset used for this work.

The 3D structures (.pdb files) were used as input for for a pre-trained graph neural nerwork, [ESM-IF1](https://github.com/facebookresearch/esm#invf) described as <code>GVPTransformer</code> in [Learning inverse folding from millions of predicted structures. (Hsu et al. 2022)](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2). This was done to obtain feature vectors capturing the geometric information of the input data. To obtain feature vectors for all proteins of the dataset instructions described in section [Encoder output as structure representation](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding) was followed. The necessary functions needed to obtain the feature vectors is shown in the cell below.  

```python
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
```
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
### Output files
Five models from the 5-fold cross-validation setup
- model.1.pt
- model.2.pt
- model.3.pt
- model.4.pt
- model.5.pt

### Use models to predict topology 
Follow instructions in the [scripts/Test.ipynb]() notebook 

