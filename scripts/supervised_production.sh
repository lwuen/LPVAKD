python ../src/train_teacher_gnn.py --datasets=cora --encoder=gcn --runs=10 --transductive=production
python ../src/train_teacher_gnn.py --datasets=citeseer --encoder=gcn --runs=10 --transductive=production
python ../src/train_teacher_gnn.py --datasets=pubmed --encode=gcn --runs=10 --transductive=production
python ../src/train_teacher_gnn.py --datasets=coauthor-cs --encoder=gcn --runs=10 --transductive=production
python ../src/train_teacher_gnn.py --datasets=coauthor-physics --encoder=gcn --runs=10 --transductive=production --hidden_channels=256
python ../src/train_teacher_gnn.py --datasets=amazon-computers --encoder=gcn --lr=0.001 --runs=10 --transductive=production
python ../src/train_teacher_gnn.py --datasets=amazon-photos --encoder=gcn --lr=0.001 --runs=10 --transductive=production
