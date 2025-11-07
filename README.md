# scGSDR: Harnessing Gene Semantics for Single-Cell Pharmacological Profiling
<img src="./scGSDR_figure.png" width="900">

We developed a method called scGSDR based on Python, which utilizes `torch` to learn the relationship between gene expression and drug response from either scRNA-seq or bulk RNA-seq data, and applies this knowledge to predict drug responses at scRNA-seq data. This project is a simple implementation of scGSDR.

## Requirements
### Python requirements
+ Python >= 3.9
+ torch >= 1.11.0
+ rpy2 >= 3.5.10

### R requirements
+ R >= 4.2.2
+ GSEABase >= 1.60.0
+ AUCell >= 1.20.2
+ SingleCellExperiment >= 1.20.1

## Usage
Run the model by executing the start_scGSDR.py file.

## Sample Dataset
You can download the sample dataset and the pathway files required to run the model from Google Drive at the following link: https://drive.google.com/drive/folders/1cE-F02Kzjczsp1Dn-wDVy3RTfhbUoNCb?usp=drive_link

After downloading the files, place the data obtained from the cloud drive into the corresponding folders within the project's root directory.

## Availability of Datasets

| Experiment | Related Drug | Source | Download Link |
| ----------- | ----------- | ----------- | ----------- |
| Experiment 1-1   | Afatinib    | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-2   | AR-42       | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-3   | Cetuximab   | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-4   | Etoposide   | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-5   | Gefitinib   | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-6   | NVP-TAE684  | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-7   | PLX4720     | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-8   | PLX4720     | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-9   | Sorafenib   | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 1-10  | Vorinostat  | [Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202204113) | [Github](https://github.com/CompBioT/SCAD/tree/main/data)
| Experiment 2-1   | PLX4720     | [GSE108383](https://pubmed.ncbi.nlm.nih.gov/30061114/)/ [GSE108394](https://pubmed.ncbi.nlm.nih.gov/30061114/) | [Google Drive](https://drive.google.com/drive/folders/19J3NMOrUVcK-gYQTPtjExHNMkdzamlH-?usp=sharing)
| Experiment 2-2   | PLX4720     | [GSE108383](https://pubmed.ncbi.nlm.nih.gov/30061114/) | [Google Drive](https://drive.google.com/drive/folders/16RG512MsYdQB11I7carAOBkJxS4qVrPV?usp=sharing)
| Experiment 2-3   | Paclitaxel/ Atezolizumab | [GSE169246](https://pubmed.ncbi.nlm.nih.gov/34653365/) | [Google Drive](https://drive.google.com/drive/folders/1gVDc751bEsrmKyL2Pyl5PUwDmOFoHBKU?usp=sharing)
| Experiment 2-4   | Paclitaxel  | [GSE169246](https://pubmed.ncbi.nlm.nih.gov/34653365/) | [Google Drive](https://drive.google.com/drive/folders/1saCsv0OnHX1PQKjObszC1fTw218C7goS?usp=sharing)


