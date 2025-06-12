

# Prompting without Panic: Attribute-aware, Zero-shot, Test-Time Calibration (ECMLPKDD 2025)

This repository provides the official implementation of our ECML 2025 paper:
> Prompting without Panic: Attribute-aware, Zero-shot, Test-Time Calibration    
> Authors: Ramya Hebbalaguppe*, Tamoghno Kandar*, Abhinav Nagpal, Chetan Arora

The implementation is built upon [TPT](https://github.com/azshue/TPT).



## Installation
```bash
# Clone this repo
git clone https://github.com/rhebbalaguppe/TCA_PromptWithoutPanic
cd TCA

# Create a conda enviroment
1. conda env create -f environment.yml
2. conda activate tca
```

## Datasets
Our evaluation focuses on 

1) fine-grained classification: ImageNet, Flower102, OxfordPets, SUN397, DTD, Food101, StanfordCars, Aircraft, UCF101, EuroSAT, Caltech101

2) natural distribution shift: ImageNet-V2, ImageNet-A, ImageNet-R, ImageNet-Sketch

Prepare the datasets based on the following link https://github.com/azshue/TPT.

## Running Experiments

In each of the .sh files, change the {data_root} accordingly. You can change the CLIP architecture by modifying the {arch} parameter to either ‘RN50’ or ‘ViT-B/16’. In addition, if you only want to run on a single GPU, remove the --multi_gpu flag from the last line of the respective .sh file that you run.



1. Hard Prompt Initialization
```bash
#for Fine-grained classification
bash scripts/test_tpt_tca_fg.sh {dataset} {num_attributes}

#for natural distribution shift
bash scripts/test_tpt_tca_ds.sh {dataset} {num_attributes}

#for temperature scaling experiments, change the run_type to tpt_ts in the .sh file.
```

2. Ensemble Initialization
```bash
#for Fine-grained classification
bash scripts/test_tpt_tca_fg_ensemble.sh {dataset} {num_attributes}

#for natural distribution shift
bash scripts/test_tpt_tca_ds_ensemble.sh {dataset} {num_attributes}
```
The command line argument {dataset} can be specified as follows: ‘I’, ‘DTD’, ‘Flower102’, ‘Food101’, ‘Cars’, ‘SUN397’, ‘Aircraft’, ‘Pets’, ‘Caltech101’, ‘UCF101’, or ‘eurosat’ for fine-grained classification datasets, and ‘V2’, ‘A’, ‘R’, or ‘K’ for datasets with natural distribution shifts. Depending on the dataset and the number of attributes, you have to change the last 2 parameters.
