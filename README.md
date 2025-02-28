IASGAN: Ionogram Automatic Scaling with Adversarial Learning

by Wen Liu, Xinxin Huang, Na Wei, and Zhongxin Deng

Code for paper 'Automatic Scaling of Vertical Ionograms Based on Generative Adversarial Network'

Dataset

link to dataset: http://www.geophys.ac.cn/ArticleDataInfo.asp?MetaId=205.

Prerequisites

Python3.9
Tensorflow2.8
Segmentation-Models
PyYaml

Training

You can use --gpu argument to specifiy gpu. To train a model, first create a configuration file (see example_config.yaml) Then run

python train.py

Testing

To test, run

python test.py

Evaluation

You can evaluate the model's performance by running script:

python eval.py
