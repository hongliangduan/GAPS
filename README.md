# GAPS: Geometric Attention-based Networks for Peptide Binding Sites Identification by the Transfer Learning Approach

![overviwe](util/overview.png)

GAPS demonstrates state-of-the-art (SOTA) performance in the protein-peptide binding sites prediction which also exhibits exceptional performance across several expanded experiments including predicting the apo protein-peptide, the protein-cyclic peptide, and the AlphaFold-predicted protein-peptide binding sites, revealing that the GAPS is a powerful, versatile, stable method suitable for diverse binding site predictions

## Installation

Download the source code and install the dependencies:

```bash
git clone https://github.com/hongliangduan/GAPS.git
cd GAPS
conda create -n gaps python=3.10
conda activate gaps
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pandas scikit-learn tqdm h5py gemmi 
```

## Usage

## Prediction

If you just want to predict the peptide binding sites on a protein, you can use the [prediction.ipynb](prediction.ipynb) by specifying your own PDB files path and the prediction results are stored in the same folder. You can use the [PyMOL](https://pymol.org) to visualize them with,

```bash
spectrum b
```

The confidence scores of the binding sites are colored by the gradient from the blue to the red. In a word, more red more confident.

## Training

· But if you want to train a new model by youself, you must download the PDB files fistly. You can use the [download_pepnn_data.py](download_pepnn_data.py) to download the protein-peptide complexes which ues to fine-tune the model and ues the [download_scannet_data.py](download_scannet_data.py) to download the protein-protein complexes which ues to pre-training the model.

· After downloading the all data, you can use the [build_dataset.py](build_dataset.py) to construct the dataset for training. Based on the type of complex, you can construct the protein-protein binding sites dataset or protein-peptide binding sites dataset.

· Then you can use the [main.py](main.py) to train a new model.

## Reference

**If you find this code and idea useful in your research, please consider citing:**
Zhu, C.; Zhang, C.; Shang, T.; Zhang, C.; Zhai, S.; Su, Z.; Duan, H. GAPS: Geometric Attention-Based Networks for Peptide Binding Sites Identification by the Transfer Learning Approach. bioRxiv December 26, 2023, p 2023.12.26.573336. [https://doi.org/10.1101/2023.12.26.573336](https://doi.org/10.1101/2023.12.26.573336)
