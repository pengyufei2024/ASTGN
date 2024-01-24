Network Traffic Prediction with Attention-based Spatial-Temporal Graph Network


<div align=center>
<img src="https://github.com/pengyufei2024/ASTGN/blob/main/figure/ASTGN.png" width="50%">
</div>

# Requirements
see requirements.txt


# Data Preparation
1.Download the network traffic matrix dataset and generate the .h5 dataset file using 'GenerateDataset.py', the .h5 file should be placed in the `data/` folder.

```
python GenerateDataset.py
```

2.Generate SE.txt using the code in the node2vec folder. The SE.txt file should also be placed in the `data/` folder.


# Model Training
After setting the parameters in train.py, execute the following commands to train the model:

```
python train.py
```


