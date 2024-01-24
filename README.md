![image](https://github.com/pengyufei2024/ASTGN/assets/157557824/6fac0600-8c35-4bf3-8e1d-93bf0e2bee17)![image](https://github.com/pengyufei2024/ASTGN/assets/157557824/a9e73cae-650e-4799-b535-c90aa78634c6)# ASTGN
Network Traffic Prediction with Attention-based Spatial-Temporal Graph Network

<img src="https://github.com/pengyufei2024/ASTGN/blob/main/figure/ASTGN.png" width="50%">

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


