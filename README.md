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
(If you are not familiar with python, we suggest you to use Anaconda to install these prerequisites.)

Training

You can use --gpu argument to specifiy gpu. To train a model, first create a configuration file (see example_config.yaml) Then run

python train.py

Tips: According to feedback that certain implementations of RAdam optimizer have problems in training convergence in this program, switch to Adam optimizer can solve the problem.

Testing

To test, run

python test.py

Evaluation

You can evaluate the model's performance by running script:

python eval.py
