python ../src/main.py --datasets=cora  --LPVAKD_D=0.001  --LPVAKD_R=1 --True_label=0.1 --dropout=0.5 --encoder=gcn --hop=2 --hops=2 --lr=0.01 --margin=0.1 --ns_rate=1 --ps_method=nb --rw_step=3 --transductive=transductive
python ../src/main.py --datasets=citeseer  --LPVAKD_D=0.001 --LPVAKD_R=1 --True_label=0.1 --dropout=0.5 --encoder=gcn --hop=2 --hops=1 --lr=0.01 --margin=0.1 --ns_rate=4 --ps_method=nb --rw_step=3 --transductive=transductive
python ../src/main.py --datasets=pubmed  --LPVAKD_D=0.1 --LPVAKD_R=0.1 --True_label=0.0001 --dropout=0.0 --encoder=gcn --hop=2 --hops=3 --lr=0.01 --margin=0.05 --ns_rate=5 --ps_method=nb --rw_step=5 --transductive=transductive
python ../src/main.py --datasets=coauthor-cs --LPVAKD_D=100 --LPVAKD_R=0.1 --True_label=10 --dropout=0.0 --encoder=gcn --hop=2 --hops=3 --lr=0.001 --margin=0.1 --ns_rate=4 --ps_method=nb --rw_step=3 --transductive=transductive
python ../src/main.py --datasets=coauthor-physics  --LPVAKD_D=1  --LPVAKD_R=0 --True_label=1 --dropout=0.0 --encoder=gcn --hidden_channels=512 --hop=2 --hops=15 --lr=0.001 --margin=0.05 --ns_rate=3 --num_layers=2 --ps_method=nb --rw_step=1  --transductive=transductive
python ../src/main.py --datasets=amazon-photos  --LPVAKD_D=1  --LPVAKD_R=1 --True_label=1 --dropout=0.0 --encoder=gcn --hidden_channels=256 --hop=2 --hops=15 --lr=0.001 --margin=0.05 --ns_rate=10 --num_layers=2 --ps_method=nb --rw_step=3 --transductive=transductive
python ../src/main.py --datasets=amazon-computers  --LPVAKD_D=1  --LPVAKD_R=1 --True_label=1 --dropout=0.0 --encoder=gcn --hidden_channels=512 --hop=2 --hops=5 --lr=0.0005 --margin=0.05 --ns_rate=5 --num_layers=2 --ps_method=nb --rw_step=2 --transductive=transductive
python ../src/main.py --datasets=collab  --LPVAKD_D=1  --LPVAKD_R=0 --True_label=1 --dropout=0.0 --encoder=gcn --hidden_channels=1024 --hop=2 --hops=3 --lr=0.001 --margin=0.01 --ns_rate=3 --num_layers=3 --ps_method=nb --rw_step=3  --transductive=transductive



