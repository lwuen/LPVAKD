import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from utils import get_dataset, do_edge_split
from torch_sparse import SparseTensor
from os.path import exists
from matrix_convert_table import *
import argparse
def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--link_batch_size', type=int, default=16 * 1024)
    parser.add_argument('--node_batch_size', type=int, default=16 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--datasets', type=str, default='cora')
    parser.add_argument('--predictor', type=str, default='mlp', choices=['inner', 'mlp'])
    parser.add_argument('--patience', type=int, default=200, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@20', choices=['auc', 'hits@20', 'hits@50'],
                        help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--True_label', default=0.1, type=float, help="true_label loss")
    parser.add_argument('--KD_RM', default=0, type=float, help="Representation-based matching KD")
    parser.add_argument('--KD_LM', default=0, type=float, help="logit-based matching KD")
    parser.add_argument('--LLP_D', default=1, type=float, help="distribution-based matching kd")
    parser.add_argument('--LLP_R', default=1, type=float, help="rank-based matching kd")
    parser.add_argument('--margin', default=0.1, type=float, help="margin for rank-based kd")
    parser.add_argument('--rw_step', type=int, default=3, help="nearby nodes sampled times")
    parser.add_argument('--ns_rate', type=int, default=1, help="randomly sampled rate over # nearby nodes")
    parser.add_argument('--hops', type=int, default=2, help="random_walk step for each sampling time")
    parser.add_argument('--hop', type=int, default=2, help="random_walk step for each sampling time")
    parser.add_argument('--ps_method', type=str, default='nb',
                        help="positive sampling is rw or nb")  # rw是random walk nb是
    parser.add_argument('--transductive', type=str, default='transductive', choices=['transductive', 'production'])
    parser.add_argument('--minibatch', action='store_true')
    parser.add_argument('--agg_feature', action='store_false')
    parser.add_argument('--use_new_feature', action='store_false')
    parser.add_argument('--use_discriminator', action='store_false')
    parser.add_argument('--discriminator_lr', type=float, default=0.005)
    parser.add_argument('--ad', default=1, type=float, help="")
    parser.add_argument('--G', default=1, type=float, help="")


    args = parser.parse_args()
    Logger_file = "../results/" + args.datasets + "_KD_" + args.transductive + ".txt"
    file = open(Logger_file, "a")
    file.write(str(args) + "\n")
    if args.KD_RM != 0:
        file.write("Logit-matching\n")
    elif args.KD_LM != 0:
        file.write("Representation-matching\n")
    elif args.LLP_D != 0 or args.LLP_R != 0:
        file.write("LLP (Relational Distillation)\n")
    file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    mini_batch_device = 'cpu'

    ### Prepare the datasets
    if args.transductive == "transductive":
        if args.datasets != "collab":
            dataset = get_dataset(args.dataset_dir, args.datasets)
            data = dataset[0]

            if exists("../data/" + args.datasets + ".pkl"):
                split_edge = torch.load("../data/" + args.datasets + ".pkl")
            else:
                split_edge = do_edge_split(dataset)
                torch.save(split_edge, "../data/" + args.datasets + ".pkl")

            edge_index = split_edge['train']['edge'].t()
            data.adj_t = edge_index
            input_size = data.x.size()[1]
            args.metric = 'Hits@20'

        elif args.datasets == "collab":
            dataset = PygLinkPropPredDataset(name='ogbl-collab')
            data = dataset[0]
            edge_index = data.edge_index
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            data = T.ToSparseTensor()(data)

            split_edge = dataset.get_edge_split()
            input_size = data.num_features
            args.metric = 'Hits@50'
            data.adj_t = edge_index

        # Use training + validation edges for inference on test set.
        if args.use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
            if args.datasets != "collab":
                data.full_adj_t = full_edge_index
            elif args.datasets == "collab":
                data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
                data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t

        if args.minibatch:
            data = data.to(mini_batch_device)
        else:
            data = data.to(args.device)

        if args.agg_feature:
            os.makedirs("../saved_agg_features", exist_ok=True)
            for _ in range(args.hop):
                data.x = aggregate_node_feature(data.x, data.full_adj_t)
            torch.save(data.x,"../saved_agg_features/" + args.datasets + "_" + args.encoder + "_" + args.transductive + "_data" + str(args.hop) + ".pkl")
    else:
        training_data, val_data, inference_data, data, test_edge_bundle, negative_samples, _ = torch.load("../data/" + args.datasets + "_production.pkl")
        training_data.to(args.device)
        for _ in range(args.hop):
            training_data.x = aggregate_node_feature(training_data.x, training_data.edge_index)
        val_data.to(args.device)
        inference_data.to(args.device)
        for _ in range(args.hop):
            val_data.x = aggregate_node_feature(val_data.x, val_data.edge_index)
            inference_data.x = aggregate_node_feature(inference_data.x, inference_data.edge_index)
        torch.save(training_data.x, "../sava_agg_features/" + args.datasets + "_" + args.encoder + "_" + args.transductive + "_training_data" +  str(args.hop) + ".pkl")
        torch.save(val_data.x, "../sava_agg_features/" + args.datasets + "_" + args.encoder + "_" + args.transductive + "_val_data" +  str(args.hop)  + ".pkl")
        torch.save(val_data.x, "../sava_agg_features/" + args.datasets + "_" + args.encoder + "_" + args.transductive + "_inference_data" + str(args.hop) + ".pkl")
    print("over")
main()

