# Active Learning in Computational Pathology with Noise Detection Empowered by Loss-Based Prior and Feature Analysis

a histopathology image active learning framework (BPAL) capable of dynamically identifying noisy labeled samples and images requiring annotation.



# Abstract

The analysis of histopathological images is pivotal in the diagnosis and treatment of cancer. However, the manual annotation of extensive image datasets is both costly and time-consuming. Moreover, sample labeling demands specialized expertise and often exhibits significant inter-annotator variability among pathologists, leading to noisy datasets. This study introduces BPAL (Beta Mixture Model and Penalized Regression for Active Learning), a novel active learning framework designed for pathological image analysis. BPAL aims to reduce expert annotation costs and mitigate the impact of noisy samples during training by autonomously managing highly informative samples in each active learning iteration. Our approach integrates two noise detection modules into active learning frameworks. By incorporating Penalized Regression (PR) with parallel computation capabilities into our framework, we enhance the efficiency of noisy sample detection. Leveraging a Beta Mixture Model (BMM) with prior loss knowledge further augments this process by enabling a comprehensive analysis from various angles within the merged feature and label spaces. This approach maximizes the utilization of information extracted from pathological image samples, ensuring a robust and thorough assessment of data quality. We propose a heuristic sampling strategy based on these enhancements. High-information samples identified by the module are categorized into three types: typical samples with high confidence levels that can receive pseudo labels for training, difficult samples requiring expert re-annotation due to complex features, and mislabeled noisy samples. The iterative addition of training sets retains high-information samples while mitigating the impact of noisy samples. Comparative evaluations demonstrate the superior performance of our approach on breast cancer and prostate cancer classification tasks.



# Data structure

## Download

### BCSS

You can download the data necessary to use the present code and reproduce our results here:

- raw_data: [Google Drive](https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss)
- preprocessed_data: [Google Drive](https://drive.google.com/drive/folders/1jVWxTae4hftTAKBwPeLZ4NXjcA9I0QBw?usp=drive_link)
- weights:[Google Drive](https://drive.google.com/drive/folders/1Uqp1rzAxOdFwxYHLLyu3gdOyzU2qjHsE?usp=drive_link)

### PANDA

You can download the data from [kaggle](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data)

## Description

### BCSS

```txt
raw_data/
├── meta : details for each image and download config
├── masks : where the ground truth masks to use for training and validation are saved
├── rgbs_colorNormalized: where RGB images corresponding to masks are saved
├── logs : in case anythign goes wrong

preprocessed_data/
├── init_labeled_set.npy : initial labeled images id set
├── init_unlabeled_set.npy : initial unlabeled images id set
├── test_dataset.npy : images for testing
├── train_dataset.npy : images for training
├──	patch_file.npy ：information for train patches
├── ts_patch_file.npy : information for test patches
├── bk_dataset.npy :  images all patches
├── bk_patch_file.npy : information for all patches

weights/
├── BPAL-BCSS.pth.tar : pretrained weights of Resnet-50 under the BPAL framework
├── PathAL-BCSS.pth.tar : Pretrained weights of Resnet-50 under the PathAL framework
```

### PANDA

**[train/test].csv**

- `image_id`: ID code for the image.
- `data_provider`: The name of the institution that provided the data. Both the [Karolinska Institute](https://ki.se/en/meb) and [Radboud University Medical Center](https://www.radboudumc.nl/en/research) contributed data. They used different scanners with slightly different maximum microscope resolutions and worked with different pathologists for labeling their images.
- `isup_grade`: Train only. The target variable. The severity of the cancer on a 0-5 scale.
- `gleason_score`: Train only. An alternate cancer severity rating system with more levels than the ISUP scale. For details on how the gleason and ISUP systems compare, see the [Additional Resources tab](https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview/additional-resources).

**[train/test]_images**: The images. Each is a large multi-level tiff file. You can expect roughly 1,000 images in the hidden test set. Note that slightly different procedures were in place for the images used in the test set than the training set. Some of the training set images have stray pen marks on them, but the test set slides are free of pen marks.

**train_label_masks**: Segmentation masks showing which parts of the image led to the ISUP grade. Not all training images have label masks, and there may be false positives or false negatives in the label masks for a variety of reasons. These masks are provided to assist with the development of strategies for selecting the most useful subsamples of the images. The mask values depend on the data provider:

Radboud: Prostate glands are individually labelled. Valid values are:

0: background (non tissue) or unknown

1: stroma (connective tissue, non-epithelium tissue)

2: healthy (benign) epithelium

3: cancerous epithelium (Gleason 3)

4: cancerous epithelium (Gleason 4)

5: cancerous epithelium (Gleason 5)

Karolinska: Regions are labelled. Valid values are:

0: background (non tissue) or unknown

1: benign tissue (stroma and epithelium combined)

2: cancerous tissue (stroma and epithelium combined)



# `BPAL` 

## Hardware

As a pre-requirement, we suggest to work on a machine with at least 8 CPUs, RTX 3080 Ti / 12 GB. 

## Installation

### Installing `bpal` package within this repo

Create a dedicated conda environment (optional):

```
conda create -n bpal python=3.11
conda activate bpal
```

Once the installation and data download steps are completed, you finally need to edit the `./config/config.yaml` file so that to specify:

- `dataset/path`: the path where preprocessed_data downloaded 
- `save_data_dir`: the dir where result will save
- `noise_rate` : noise rate
- `pretrained_iteration` : epoch for each iteration
- `max_iteration `: number of iteration



### Run tests

Once data has been downloaded and the previous installation steps done, you can run the full test suite to make sure features are loaded correctly. You first need to add specific requirements via:

```
python -m pip install -r requirements.txt
```

or

```ssh
bash install.sh
```



Then, you can run the whole stack of tests by running the following command 

```ssh
python main.py
```
