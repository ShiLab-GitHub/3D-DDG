# Author statement

This repository contains supporting files of data and code for the paper manuscript titled "3D-ΔΔG: A dual-channel prediction model for protein-protein binding affinity changes following mutation based on protein 3D structures". The focus of this work is to predict changes in protein-protein affinity using three-dimensional structural information and amino acid sequences of variant side chains.

# Resources:

+ README.md: this file.
+ data/pdb_graph：This folder is used to save the results of the graph characterization code run out.(Create your own)
+ seq_embs/prot_aibert：This folder holds the results of generating embedding codes for protein sequences.
+ my_model/aibert：Used to store the configuration file for the protaibert pre-trained large model, which is also the pre-trained model used in this project(Can be found at https://figshare.com/s/8bbfb73e8e47d4312911 )
+ my_model/output:  Holds the three result plots of the training output(Create your own)
+  my_model/SKEMPI_all_pdbs: Where the pdb files used in this project are saved, this is the one used in this project.(Can be found at https://figshare.com/s/8bbfb73e8e47d4312911 )

###  Source codes:

+ config.py: model configuration file
+ my_dataset.py: customized dataset class for torch
+ model.py: model file
+  process_data_seq.py:code for generating sequence embedding files
+ process_data_stru_muti.py:after each pdb file generates the graph characterization individually, the simultaneous multi-process generation of the graph characterization files, the other files starting with process_data_stru are the unused code for generating the graph characterization used before.
+ protseqfeature.py/utils.py:reference files needed for training code
+ training.py:code used to train the model

# Step-by-step running:

## 0. Install Python libraries needed

+ Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
+ Install Transformer: conda install -c huggingface transformers
+ Or run the following commands to install both pytorch_geometric and Transformer:

```
conda create -n geometric python=3
conda activate geometric
conda install -c huggingface transformers
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric

```

## 1. Selection of datasets

In the . /data/sample folder, there are the training and validation sets that we processed. They belong to three datasets, SKP1102s, SKEMPIv1, and SKEMPIv2. Before you start the project running, pick one of the three folders according to your experimental needs and copy the skempi_train.csv and skempi_test.csv files inside to the .data/ folder directory.

## 2. Generating embedding using pre-trained macromodels

Run the process_data_seq.py code to generate the embedding encoding results. Two files are generated after the run, seq2path_prot_albert.pickle and \data\seq_embs\prot_albert.

## 3. Protein 3D structure map characterization was generated

Run the process_data_stru_muti.py file to generate 3D structure map characterization results. Two files are generated after the run, pdb_stru_graph.pickle and data\pdb_graph.

## 4. Train a prediction model

Run the training.py file to generate a csv file of the results of the corresponding predictions, a weights file for the model, and the corresponding results plots.
