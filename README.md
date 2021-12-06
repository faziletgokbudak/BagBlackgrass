# DSMIL: Dual-stream multiple instance learning networks
This is the Pytorch implementation for the multiple instance learning model described in the paper [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939) (_CVPR 2021, accepted for oral presentation_).  

<div align="center">
  <img src="thumbnails/overview.png" width="700px" />
</div>

<div align="center">
  <img src="thumbnails/overview-2.png" width="700px" />
</div>

## Installation
Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)  
Required packages
```
  $ conda env create --name dsmil --file env.yml
  $ conda activate dsmil
```
Install [OpenSlide and openslide-python](https://pypi.org/project/openslide-python/).  
[Tutorial 1](https://openslide.org/) and [Tutorial 2 (Windows)](https://www.youtube.com/watch?v=0i75hfLlPsw).  

Useful arguments:
```
[--num_classes]       # Number of non-negative classes
[--feats_size]        # Size of feature vector (depends on the CNN backbone)
[--lr]                # Initial learning rate [0.0002]
[--num_epochs]        # Number of training epochs [200]
[--weight_decay]      # Weight decay [5e-3]
[--dataset]           # Dataset folder name
[--split]             # Training/validation split [0.2]
[--dropout_patch]     # Randomly dropout a portion of patches and replace with duplicates during training [0]
[--dropout_node]      # Randomly dropout a portion of nodes in the value vector generation network during training [0]
```
## Training on your own datasets
1. Place the original images in datasets 
```
e.g.
    /mnt/yifan/data/blackgrass/blackgrass/
```
2. Crop patches.  
```
  $ python deepzoom_tiler.py -m 0 -b 20 -d [DATASET_NAME] -l [LIST_PATH]
e.g. 
   python deepzoom_tiler.py -m 0 -b 20 -d /mnt/yifan/data/blackgrass/ -l /mnt/yifan/data/blackgrass/blackgrass/data_table.txt
The output is a folder called 'single' 
e.g.
    /mnt/yifan/data/blackgrass/single/
            --no_blackgrass
            --sparse_blackgrass
The classes will depend on the classes in data_table.txt
```
>Set flag `-m [LEVEL 1] [LEVEL 2]` to crop patches from multiple magnifications. 
3. Train an embedder.  
```
  $ cd simclr
  $ python run.py --dataset=[DATASET_NAME]
  e.g.
    python run.py --dataset=/mnt/yifan/data/blackgrass/

The trained model will be saved at YOUR_CODE_FOLDER/simclr/runs/TIME_TAG/checkpoints
e.g.
    /mnt/share/yifan/code/dsmil-wsi/simclr/runs/Dec01_15-44-20_JHCDT/checkpoints/model.pth
```
>See YOUR_CODE_FOLDER/simclr/config.yaml
>  Modified root_dir, list_dir and input_c, accordingly
> input_c could be 3 4 5, following the order in 'data_table.txt'
>Set flag `--multiscale=1` and flag `--level=low` or `--level=high` to train an embedder for each magnification if the patches are cropped from multiple magnifications.   
4. Compute features using the embedder.  
```
  $ cd ..
  $ python compute_feats.py --dataset=[DATASET_NAME] --input_c C
  e.g.
    python compute_feats.py --dataset=/mnt/yifan/data/blackgrass/ --input_c 5
```
>The feature will be saved in "[DATASET_NAME]/features/"
>You will get bags_all.csv, CLASSNAME.csv for training.
>Set flag `--magnification=tree` to compute the features for multiple magnifications.
>This will use the last trained embedder to compute the features, if you want to use an embedder from a specific run, add the option `--weights=[RUN_NAME]`, where `[RUN_NAME]` is a folder name inside `simclr/runs/`. If you have an embedder you want to use, you can place the weight file as `simclr/runs/[RUN_NAME]/checkpoints/model.pth` and pass the `[RUN_NAME]` to this option. To use a specific embedder for each magnification, set option `--weights_low=[RUN_NAME]` (embedder for low magnification) and `--weights_high=[RUN_NAME]` (embedder for high magnification). The embedder architecture is ResNet18 with **instance normalization**.     

5. Training.
```
  $ python train_tcga.py --dataset=[DATASET_NAME]/features
e.g.
    python train_tcga.py --dataset=/mnt/yifan/data/blackgrass/features
```
>You will need to adjust `--num_classes` option if the dataset contains more than 2 positive classes or only 1 positive class and 1 negative class (binary classifier). See the next section for details.  

6. Testing.
```
  $ python attention_map.py --bag_path test/patches --map_path test/output --thres DEPEND_ON_TRAINING ----embedder_weights /mnt/yifan/data/blackgrass/embedder.pth --aggregator_weights TRAINED_WEIGHT
```

Useful arguments:
```
[--num_classes]         # Number of non-negative classes.
[--feats_size]          # Size of feature vector (depends on the CNN backbone).
[--thres]               # List of thresholds for the classes returned by the training function.
[--embedder_weights]    # Path to the embedder weights file (saved by SimCLR). Use 'ImageNet' if ImageNet pretrained embedder is used.
[--aggregator_weights]  # Path to the aggregator weights file.
[--bag_path]            # Path to a folder containing folders of patches.
[--patch_ext]            # File extensino of patches.
[--map_path]            # Path of output attention maps.
```
## Feature vector csv files explanation
1. For each bag, there is a .csv file where each row contains the feature of an instance. The .csv is named as "_bagID_.csv" and put into a folder named "_dataset-name_/_category_/".  

<div align="center">
  <img src="thumbnails/bag.png" width="700px" />
</div>  

2. There is a "_dataset-name_.csv" file with two columns where the first column contains the paths to all _bagID_.csv files, and the second column contains the bag labels.  

<div align="center">
  <img src="thumbnails/bags.png" width="700px" />
</div>  

3. Labels.
> For binary classifier, use `1` for positive bags and `0` for negative bags. Use `--num_classes=1` at training.  
> For multi-class classifier (`N` positive classes and one optional negative class), use `0~(N-1)` for positive classes. If you have a negative class (not belonging to any one of the positive classes), use `N` for its label. Use `--num_classes=N` (`N` equals the number of **positive classes**) at training.

