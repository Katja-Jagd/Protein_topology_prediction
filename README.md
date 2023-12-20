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
```python
git clone https://github.com/Katja-Jagd/Protein_topology_prediction
```
## Train model
```python
usage: python main.py [-h] [-i INPUTS] [--epochs N] [--lr LR] [-b N] [--mode {train,test}]
```



