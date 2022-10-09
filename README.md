# Domain Adaptation via Surrogate Mutual Information Maximization 


## Introduction

This is the code of our work [Domain Adaptation via Surrogate Mutual Information Maximization](https://www.ijcai.org/proceedings/2022/0237.pdf) published on IJCAI 2022.
<div align=center>
<img src="https://github.com/zhao-ht/SIDA/blob/master/fig1.png" width="600px">
</div>


To reproduce our experiments, please first install conda environment and download datasets.




## Environment



```
conda create -n sida python=3.6
source activate sida

pip install -r requirements.txt
```

## Datasets


```
cd experiments
cd dataset

# download Office-31 from https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA
# download Office-Home from https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw
# download VisDA2017 from
# https://drive.google.com/file/d/0BwcIeDbwQ0XmdENwQ3R4TUVTMHc/view?usp=sharing
# https://drive.google.com/file/d/0BwcIeDbwQ0XmUEVJRjl4Tkd4bTA/view?usp=sharing
# https://drive.google.com/file/d/0BwcIeDbwQ0XmdGttZ0k2dmJYQ2c/view?usp=sharing

bash data_preprocess.sh

cd ..
cd ..
```

## Run Experiments

Run train.py with arguments --cfg to the configuration file and --exp_name for the output directory. The experiment log file and the  checkpoints will be stored at ./experiments/ckpt/${experiment_name}



For example, to run experiments of OfficeHome: 

```
CUDA_VISIBLE_DEVICES=0  python train.py --cfg ./experiments/config/OfficeHome/office31_train_A2C_cfg.yaml  --exp_name officehome_a2c
```

for VisDA-2017:

```
CUDA_VISIBLE_DEVICES=0  python train.py --cfg ./experiments/config/OfficeHome/visda17_train_train2val_cfg.yaml  --exp_name visda2017 
```

and for Office-31: 

```
CUDA_VISIBLE_DEVICES=0  python train.py --cfg experiments/config/Office-31/office31_train_amazon2dslr_cfg.yaml  --exp_name office31_a2d 
```


## Acknowledgement
The clustering framework of our code references <https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation>.







