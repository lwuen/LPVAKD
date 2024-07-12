**# Link Prediction via Adversarial Knowledge Distillation and Feature Aggregation

## Requirements
Please run the following code to install all the requirements:
```
pip install -r requirements.txt
```
## Dataset Download
When the program runs, the dataset is automatically downloaded

## Usage
### Transductive Setting 
- **Teacher GNN training.** You can change "gcn" to "mlp" to obtain supervised training results with MLP. 
```
python train_teacher_gnn.py --datasets=cora --encoder=gcn --transductive=transductive
```
To reproduce the supervised results shown in Table 2, you can just simply run the following command. The results will be shown in results/.
```
cd scripts/
bash supervised_transductive.sh
```
- **Student MLP training.** LPVAKD_D  indicate the weights for the distribution-based  matching KD.
```
python ../src/main.py --datasets=cora  --LPVAKD_D=0.001  --True_label=0.1   --transductive=transductive
```
To reproduce the results shown in Table 2 transductive setting, please run the following command:
```
cd scripts/
bash LPVAKD_transductive.sh
```
### Production Setting
- **Pre-process dataset**
In this work,  This setting mimics practical link prediction use cases. Under the production setting, the newly occurred nodes and edges that can not be seen during the training stage would appear in the graph at inference time. . If you want to apply this setting on our own datasets or split the datsets by your self, please change the dataset name ("dataset") in Line 194 in generate_production_split.py file and run the following command:
```
python generate_production_split.py
```
- **Teacher GNN training.** Note: changing "gcn" to "mlp" can reproduce the supervised training results with MLP.
```
python train_teacher_gnn.py --datasets=cora --encoder=gcn --transductive=production
```
To reproduce the supervised results shown in Table 2 GNN production setting, you can just simply run the following command. The results will be shown in results/.
```
cd scripts/
bash supervised_production.sh
```
- **Student MLP training.** LPVAKD_D  indicate the weights for the distribution-based  matching KD.
```
ppython ../src/main.py --datasets=cora  --LPVAKD_D=0.001  --LPVAKD_R=0.01 --transductive=production

```
To reproduce the results shown in Table 2 production setting, please run the following command:
```
cd scripts/
bash LPVAKD_production.sh
```
### Reproducing Paper Results
In our experiments, we found that the link prediction performance (when evaluated with Hits@K) of models can greatly vary even when run with the same hyperparameters. Besides, the performance of our method is sensitive to the teacher GNN. Therefore, as mentioned in our paper, we run a hyperparameter sweep for each setting and report the results from the best-performing model (as measured by validation Hits@K).

## Reference

## Contact
Please contact 6223111046@stu.jiangnan.edu.cn if you have any questions.**
