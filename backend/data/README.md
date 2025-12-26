# Backend Data Directory – TruthLens

This directory is reserved for datasets used by the TruthLens backend during
model training and evaluation.

Due to file size and licensing constraints, datasets are NOT included directly
in this GitHub repository but dataset links , format and description are provided in datset directory

---

## Dataset Used

LIAR: A Benchmark Dataset for Fake News Detection

Source:
https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

The LIAR dataset consists of short political statements labeled with different
levels of truthfulness.

For this project, the labels are converted into a binary classification task:

- Fake (0): pants-fire, false, barely-true
- Real (1): half-true, mostly-true, true

Others are 
FakeNewsNet
KaggleFakeNews
---

## Expected Directory Structure

After downloading and extracting the dataset, the directory should look like:

backend/data/liar/
- train.tsv
- valid.tsv
- test.tsv

---

## How to Download the Dataset

Run the following commands from the project root:

cd backend/data  
wget https://www.cs.ucsb.edu/~william/data/liar_dataset.zip  
unzip liar_dataset.zip  
mv liar_dataset liar  

---

## Usage in Backend

The backend training pipeline expects the dataset to be located at:

backend/data/liar

This path is passed to the training function in the backend:

train_model(data_path="backend/data/liar")

---

## Notes

- Dataset files (.tsv, .zip) are ignored using .gitignore
- This directory only stores data locally for training
- Keeping datasets out of version control ensures a clean repository
