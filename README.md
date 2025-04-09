<div align="center">

# MDDTA: a drug target binding affinity pre-diction method based on molecular dynam-ics simulation data enhancement.

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![rdkit](https://img.shields.io/badge/-rdkit_2023.3.2+-792ee5?logo=rdkit&logoColor=white)](https://anaconda.org/conda-forge/rdkit/)
[![torch-geometric](https://img.shields.io/badge/torch--geometric-2.3.1+-792ee5?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![deepchem](https://img.shields.io/badge/deepchem-2.7.1+-792ee5?logo=deepchem&logoColor=white)](https://deepchem.io/)

</div>

## 📄  Introduction 
This is a code repository for MDDTA, a deep learning based drug target binding affinity prediction model. This repository will present the data and code used for training, as well as the processes involved in drug screening.

## 🔑 Architecture
![MDDTA](https://github.com/zhaolongNCU/MDDTA/blob/main/img/MDDTA.jpg)
## 🔨 Installation
First, you need to clone our code to your operating system.

```
git clone https://github.com/zhaolongNCU/MDDTA.git
cd MDDTA
```

## 💻 The environment of PocketDTA
Before running the code, you need to configure the environment, which mainly consists of the commonly used torch==1.13.1+cu117, rdkit==2023.3.2, torch-geometric==2.3.1 and other basic packages.
```
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
Of course you can also directly use the following to create a new environment:
```
conda create -n MDDTA python==3.7
conda activate MDDTA 
pip install requirements.txt
```
where the requirements.txt file is already given in the code.
Furthermore, our code is based on python 3.7 and CUDA 11.7 and is used on a linux system. Note that while our code does not require a large amount of running memory, it is best to use more than 48G of RAM if you need to run it.
## 📚 Dataset
The MD-PDBbind dataset we constructed has been uploaded to the zenodo platform [MD-PDBbind.zip](https://zenodo.org/records/15137143) compressed file (containing sdf and pdb structure files for 10 representative dynamic conformations for each complex). In addition, the files required for training and testing have also been uploaded to the zenodo platform, namely the training set [PDBbind_2020_md.csv](https://zenodo.org/records/15137143), the test set [CASF2016.csv](https://zenodo.org/records/15137143), the protein embedding characterisation file [protein_emb_data.h5](https://zenodo.org/records/15137143), and the protein coordinates file [protein_coords_data.h5](https://zenodo.org/records/15137143), respectively. ligand embedding characterisation and coordinates file [ligand_data.h5](https://zenodo.org/records/15137143). If you want to perform retraining, simply download these five files and place them in the folder dataset that corresponds to the code.

Examples are given of how these h5 files work.This allows access to the ESM-2 embedded characterisation of the target 3ZZF_13.
```
import h5py
test = h5py.File('protein_emb_data.h5','r')
test['embedding']['3ZZF_13'][:]
```
The coordinates of the target 3ZZF_13 can be obtained by the following code:
```
import h5py
test = h5py.File('protein_coords_data.h5','r')
test['coords']['3ZZF_13'][:]
```
The coordinates and GraphMVP embedding representation of drug 3ZZF_13 can be obtained by the following code:
```
import h5py
test = h5py.File('ligand_data.h5','r')
test['coords']['3ZZF_13'][:]
test['embeddings']['3ZZF_13'][:]
```
## :gear: Pre-trained model
Since the parameter files for the other pre-trained models are rather large, we will not give them here, you can download them according to the link below and save them to the appropriate location in the MDDTA folder. [ESM-2](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt),[GraphMVP](https://github.com/chao1224/GraphMVP).

## :chart_with_upwards_trend: Training
After configuring the environment and dataset, you can run the following code for training.
```
python main.py
```

## ⏳ Drug screening
In this section we will briefly describe some of the codes used in drug screening.
![Drugbank_screen](https://github.com/zhaolongNCU/MDDTA/blob/main/img/Drugbank_screen.jpg)
### Qvina-W Molecular Docking
The files needed for molecular docking using Qvina-W are placed in the Drugbank_screen folder as follows:

**1.Install Qvina-W**
Install the molecular docking software [Qvina-W](https://qvina.github.io/):

```
git clone https://github.com/QVina/qvina.git
cd qvina
conda install -c conda-forge qvina
```
Qvina-W uses the following commands, where config.txt is the configuration file ：

```
qvina-w --config config.txt
```
**2.Generate config.txt**
Go to the drug screening folder Drugbank_screen, and run the code to generate the config.txt file:
```
cd Drugbank_screen
python config_genearation.py
```
Here in the ligand folder we put only 4 compound files, the full file [drugbank_compound.zip](https://zenodo.org/records/15137143) for drug screening has been uploaded to the zenodo platform and is available for download.
**3.Docking**
Running the following code will perform molecular docking and generate a complex file to be placed in the output folder.
```
./docking.sh
```
### MDDTA Affinity prediction
For Drugbank, the required dataset and the related file [drugbank_dataset.zip](https://zenodo.org/records/15137143) have been uploaded to zenodo, download and unzip the file, run the following code to get the compound affinity file `test_result.csv` predicted by MDDTA. 

```
python drugbank_test.py
```
