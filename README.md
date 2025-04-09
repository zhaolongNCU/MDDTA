<div align="center">

# MDDTA: A Drug Target Binding Affinity Prediction Method Based on Molecular Dynamics Simulation Data Enhancement

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![rdkit](https://img.shields.io/badge/-rdkit_2023.3.2+-792ee5?logo=rdkit&logoColor=white)](https://anaconda.org/conda-forge/rdkit/)
[![torch-geometric](https://img.shields.io/badge/torch--geometric-2.3.1+-792ee5?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/en/latest/)

</div>

## 📄 Introduction  
This repository contains the MDDTA model, a deep learning-based approach for drug target binding affinity prediction. The code and data provided facilitate training the model and conducting drug screening experiments.

## 🔑 Architecture  
![MDDTA](https://github.com/zhaolongNCU/MDDTA/blob/main/img/MDDTA.jpg)

## 🔨 Installation  
To begin, clone the repository to your local machine:

```bash
git clone https://github.com/zhaolongNCU/MDDTA.git
cd MDDTA
```

### 💻 Environment Setup  
Before running the code, set up the environment with the required dependencies. The environment consists of commonly used packages such as `torch==1.13.1+cu117`, `rdkit==2023.3.2`, `torch-geometric==2.3.1`, and others.

To set up the environment, you can create a new Conda environment:

```bash
conda create -n MDDTA python==3.7
conda activate MDDTA
pip install -r requirements.txt
```

Alternatively, install the necessary packages manually:

```bash
python==3.7.16
torch==1.13.1+cu117
torch-geometric==2.3.1
rdkit==2023.3.2
pandas==1.3.0
ogb==1.3.5
networkx==2.6.3
fair-esm==2.0.0
h5py==3.8.0
dgl==1.1.3
```

The code is compatible with Python 3.7, CUDA 11.7, and runs on Linux systems. We recommend using machines with at least 48GB of RAM for efficient execution.

## 📚 Dataset  
The MD-PDBbind dataset, which we constructed, is available on Zenodo. You can download it from the following link:  
[MD-PDBbind.zip](https://zenodo.org/records/15137143) (includes SDF and PDB structure files for 10 representative dynamic conformations for each complex).

Additional files for training and testing are also available:  
- [PDBbind_2020_md.csv](https://zenodo.org/records/15137143) (Training set)  
- [CASF2016.csv](https://zenodo.org/records/15137143) (Test set)  
- [protein_emb_data.h5](https://zenodo.org/records/15137143) (Protein embedding characterization)  
- [protein_coords_data.h5](https://zenodo.org/records/15137143) (Protein coordinates)  
- [ligand_data.h5](https://zenodo.org/records/15137143) (Ligand embedding characterization and coordinates)

Here’s how to access the ESM-2 embedded characterization of the target 3ZZF_13:

```python
import h5py
test = h5py.File('protein_emb_data.h5', 'r')
test['embedding']['3ZZF_13'][:]
```

To get the coordinates of target 3ZZF_13:

```python
import h5py
test = h5py.File('protein_coords_data.h5', 'r')
test['coords']['3ZZF_13'][:]
```

For the coordinates and GraphMVP embedding of drug 3ZZF_13:

```python
import h5py
test = h5py.File('ligand_data.h5', 'r')
test['coords']['3ZZF_13'][:]
test['embeddings']['3ZZF_13'][:]
```

## :gear: Pre-trained Models  
Due to the large size of the pre-trained model files, they are not included here. You can download them via the following links and save them to the appropriate directory in the MDDTA folder:  
- [ESM-2](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)  
- [GraphMVP](https://github.com/chao1224/GraphMVP)

## :chart_with_upwards_trend: Training  
Once the environment and dataset are set up, you can begin training the model by running:

```bash
python main.py
```

## ⏳ Drug Screening  
In this section we will briefly describe some of the codes used in drug screening.
![Drugbank_screen](https://github.com/zhaolongNCU/MDDTA/blob/main/img/Drugbank_screen.jpg)

### Qvina-W Molecular Docking  
The necessary files for molecular docking with Qvina-W are located in the `Drugbank_screen` folder.

**1. Install Qvina-W**

First, install the molecular docking software [Qvina-W](https://qvina.github.io/):

```bash
git clone https://github.com/QVina/qvina.git
cd qvina
conda install -c conda-forge qvina
```

Qvina-W uses the following command, where `config.txt` is the configuration file:

```bash
qvina-w --config config.txt
```

**2. Generate config.txt**

Navigate to the `Drugbank_screen` folder and run the code to generate the `config.txt` file:

```bash
cd Drugbank_screen
python config_generation.py
```

We place only 4 compound files in the ligand folder. The complete file for drug screening, [drugbank_compound.zip](https://zenodo.org/records/15137143), is available on Zenodo.

**3. Docking**

Run the following command to perform molecular docking and generate a complex file in the output folder:

```bash
./docking.sh
```

### MDDTA Affinity Prediction  
For DrugBank, download the dataset and associated files from Zenodo:  
[drugbank_dataset.zip](https://zenodo.org/records/15137143). After unzipping, run the following code to get the compound affinity predictions, saved in `test_result.csv`:

```bash
python drugbank_test.py
```
