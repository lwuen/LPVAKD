import argparse
import itertools
import time
import random
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import numpy as np
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger, ProductionLogger
from utils import get_dataset, do_edge_split
from models import MLP, GCN, SAGE, LinkPredictor
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists
from torch_cluster import random_walk
from torch.nn.functional import cosine_similarity
import torch_geometric
from train_teacher_gnn import test_transductive, test_production
from torch.utils.tensorboard import SummaryWriter
from matrix_convert_table import *
from Discriminator import logits_D, logits_pairs

def cosine_loss(s, t):  # 余铉相似度
    return cosine_similarity(s, t, dim=-1).mean()

def kl_loss(s, t, T):  # kl散度
    y_s = F.log_softmax(s / T, dim=-1)
    y_t = F.softmax(t / T, dim=-1)
    loss = F.kl_div(y_s, y_t, size_average=False) * (T ** 2) / y_s.size()[0]
    return loss

def neighbor_samplers(row, col, sample, x, step, ps_method, ns_rate, hops):
    batch = sample

    if ps_method == 'rw':
        pos_batch = random_walk(row, col, batch, walk_length=step * hops, coalesced=False)
    elif ps_method == 'nb':
        pos_batch = None
        for i in range(step):
            if pos_batch is None:
                pos_batch = random_walk(row, col, batch, walk_length=hops, coalesced=False)
            else:
                pos_batch = torch.cat(
                    (pos_batch, random_walk(row, col, batch, walk_length=hops, coalesced=False)[:, 1:]), 1)

    neg_batch = torch.randint(0, x.size(0), (batch.numel(), step * hops * ns_rate), dtype=torch.long)

    return pos_batch.to("cuda"), neg_batch.to("cuda")

def train(model, predictor, t_h, teacher_predictor, data, split_edge, optimizer, args, device, Discriminator_p, opt_D):
    if args.transductive == "transductive":
        pos_train_edge = split_edge['train']['edge'].to(args.device)
        row, col = data.adj_t
    else:
        pos_train_edge = data.edge_index.t()
        row, col = data.edge_index

    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    loss_dis = torch.nn.BCELoss()
    bce_loss = nn.BCELoss()  # 交叉熵损失函数
    margin_rank_loss = nn.MarginRankingLoss(margin=args.margin)  #排序损失


    total_loss = total_examples = 0
    node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
    for link_perm in DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True):

        node_perm = next(node_loader).to(args.device)

        h = model(data.x)

        edge = pos_train_edge[link_perm].t()
        loss_G = 0.0

        if args.use_discriminator:
            Discriminator_p.train()
            stu_logits = h.detach()
            tea_logits = t_h
            pos_p = Discriminator_p(tea_logits[edge[0]], tea_logits[edge[1]])
            neg_p = Discriminator_p(stu_logits[edge[0]], stu_logits[edge[1]])
            real_p = torch.sigmoid(pos_p[:, -1])
            fake_p = torch.sigmoid(neg_p[:, -1])
            p_loss = loss_dis(real_p, torch.ones_like(real_p)) + loss_dis(fake_p, torch.zeros_like(fake_p))
            ad_loss = p_loss * args.ad

            torch.nn.utils.clip_grad_norm_(tea_logits, 1.0)
            torch.nn.utils.clip_grad_norm_(stu_logits, 1.0)
            torch.nn.utils.clip_grad_norm_(Discriminator_p.parameters(), 1.0)

            opt_D.zero_grad()
            ad_loss.backward()
            opt_D.step()

            Discriminator_p.eval()

            neg_p = Discriminator_p(h[edge[0]], h[edge[1]])
            fake_p = torch.sigmoid(neg_p[:, -1])
            P_loss = loss_dis(fake_p, torch.ones_like(fake_p))

            loss_G = P_loss

        if args.LPVAKD_R or args.LPVAKD_D:
            sample_step = args.rw_step
            # sampled the neary nodes and randomely sampled nodes
            pos_sample, neg_sample = neighbor_samplers(row, col, node_perm, data.x, sample_step, args.ps_method,
                                                       args.ns_rate, args.hops)

            ### calculate the distribution based matching loss
            samples = torch.cat((pos_sample, neg_sample), 1)
            batch_emb = torch.reshape(h[samples[:, 0]], (samples[:, 0].size(0), 1, h.size(1))).repeat(1,
                                                                                                      sample_step * args.hops * (
                                                                                                                  1 + args.ns_rate),
                                                                                                      1)
            t_emb = torch.reshape(t_h[samples[:, 0]], (samples[:, 0].size(0), 1, t_h.size(1))).repeat(1,
                                                                                                      sample_step * args.hops * (
                                                                                                                  1 + args.ns_rate),
                                                                                                      1)
            s_r = predictor(batch_emb, h[samples[:, 1:]])
            t_r = teacher_predictor(t_emb, t_h[samples[:, 1:]])
            LPVAKD_d_loss = kl_loss(torch.reshape(s_r, (s_r.size()[0], s_r.size()[1])),
                                 torch.reshape(t_r, (t_r.size()[0], t_r.size()[1])), 1)

            #### calculate the rank based matching loss
            sampled_nodes = [l_i for l_i in range(sample_step*args.hops*(1+args.ns_rate))]
            dim_pairs = [x for x in itertools.combinations(sampled_nodes, r=2)]
            dim_pairs = np.array(dim_pairs).T
            teacher_rank_list = torch.zeros((len(t_r), dim_pairs.shape[1],1)).to(args.device)

            mask = t_r[:, dim_pairs[0]] > (t_r[:, dim_pairs[1]] + args.margin)
            teacher_rank_list[mask] = 1
            mask2 = t_r[:, dim_pairs[0]] < (t_r[:, dim_pairs[1]] - args.margin)
            teacher_rank_list[mask2] = -1

        if args.datasets != "collab":
            neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0), num_neg_samples=link_perm.size(0),
                                         method='dense')
        elif args.datasets == "collab":
            neg_edge = torch.randint(0, data.x.size()[0], [edge.size(0), edge.size(1)], dtype=torch.long,
                                     device=h.device)

        ### calculate the true_label loss
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(args.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        label_loss = bce_loss(out, train_label)  # 二元交叉熵损失函数


        if args.LPVAKD_D or args.LPVAKD_R:  # 交叉熵  +   余铉相似度  +  均方误差（mse)  +  kl loss  +  排序损失
            loss = args.True_label * label_loss  + args.LPVAKD_D * LPVAKD_d_loss + args.G * loss_G
        else:
            loss = args.True_label * label_loss + args.G * loss_G

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_examples = edge.size(1)
        print(cosine_loss(h[node_perm],t_h[node_perm]))

        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--link_batch_size', type=int, default=8 * 1024)
    parser.add_argument('--node_batch_size', type=int, default=8 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--datasets', type=str, default='cora')
    parser.add_argument('--predictor', type=str, default='mlp', choices=['inner', 'mlp'])
    parser.add_argument('--patience', type=int, default=200, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@20', choices=['auc', 'hits@20', 'hits@50'],
                        help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--True_label', default=0.1, type=float, help="true_label loss")

    parser.add_argument('--LPVAKD_D', default=1, type=float, help="distribution-based matching kd")
    parser.add_argument('--margin', default=0.1, type=float, help="margin for rank-based kd")
    parser.add_argument('--rw_step', type=int, default=3, help="nearby nodes sampled times")
    parser.add_argument('--ns_rate', type=int, default=1, help="randomly sampled rate over # nearby nodes")
    parser.add_argument('--hops', type=int, default=2, help="random_walk step for each sampling time")
    parser.add_argument('--hop', type=int, default=2, help="random_walk step for each sampling time")
    parser.add_argument('--ps_method', type=str, default='nb',
                        help="positive sampling is rw or nb")  # rw是random walk nb是
    parser.add_argument('--transductive', type=str, default='transductive', choices=['transductive', 'production'])
    parser.add_argument('--discriminator_lr', type=float, default=0.005)
    parser.add_argument('--ad', default=1, type=float, help="")
    parser.add_argument('--G', default=1, type=float, help="")

    parser.add_argument('--agg_feature', action='store_false')
    parser.add_argument('--use_new_feature', action='store_false')
    parser.add_argument('--use_discriminator', action='store_false')
    #
    # parser.add_argument('--agg_feature', action='store_true')
    # parser.add_argument('--use_new_feature', action='store_true')
    # parser.add_argument('--use_discriminator', action='store_true')

    args = parser.parse_args()
    print(args)

    Logger_file = "../results/" + args.datasets + "_KD_" + args.transductive + ".txt"
    file = open(Logger_file, "a")
    file.write(str(args) + "\n")
    if args.LPVAKD_D != 0 or args.LPVAKD_R != 0:
        file.write("LPVAKD (Relational Distillation)\n")
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
        agg_time = 0.0
        if args.agg_feature:
            os.makedirs("../saved_agg_features", exist_ok=True)
            if os.path.exists(
                    "../saved_agg_features/" + args.datasets + "-" + args.encoder + "_" + args.transductive + "_data" + str(
                            args.hop) + ".pkl") and not args.use_new_feature:
                data.x = torch.load(
                    "../saved_agg_features/" + args.datasets + "-" + args.encoder + "_" + args.transductive + "_data" + str(
                        args.hop) + ".pkl")
            else:
                if args.minibatch:
                    data = data.to(mini_batch_device)
                else:
                    data = data.to(args.device)
                for _ in range(args.hop):
                    data.x = aggregate_node_feature(data.x, data.full_adj_t)
                torch.save(data.x,
                           "../saved_agg_features/" + args.datasets + "-" + args.encoder + "_" + args.transductive + "_data" + str(
                               args.hop) + ".pkl")
        if args.minibatch:
            data = data.to(mini_batch_device)
        else:
            data = data.to(args.device)

        args.node_batch_size = int(data.x.size()[0] / (split_edge['train']['edge'].size()[0] / args.link_batch_size))

    else:
        training_data, val_data, inference_data, data, test_edge_bundle, negative_samples, _ = torch.load(
            "../data/" + args.datasets + "_production.pkl")
        input_size = training_data.x.size(1)

        if args.minibatch:
            training_data.to(mini_batch_device)
        else:
            training_data.to(args.device)
        if args.agg_feature:
            os.makedirs("../saved_agg_features", exist_ok=True)
            if os.path.exists(
                    "../saved_agg_features/" + args.datasets + "_" + args.encoder + "_" + args.transductive + "_training_data" + str(
                            args.hop) + ".pkl") and not args.use_new_feature:
                training_data.x = torch.load(
                    "../saved_agg_features/" + args.datasets + "_" + args.encoder + "_" + args.transductive + "_training_data" + str(
                        args.hop) + ".pkl")
                val_data.x = torch.load(
                    "../saved_agg_features/" + args.datasets + "_" + args.encoder + "_" + args.transductive + "_val_data" + str(
                        args.hop) + ".pkl")
                inference_data.x = torch.load(
                    "../saved_agg_features/" + args.datasets + "_" + args.encoder + "_" + args.transductive + "_inference_data" + str(
                        args.hop) + ".pkl")
            else:
                for _ in range(args.hop):
                    training_data.x = aggregate_node_feature(training_data.x, training_data.edge_index)
                val_data.to(args.device)
                inference_data.to(args.device)
                for _ in range(args.hop):
                    val_data.x = aggregate_node_feature(val_data.x, val_data.edge_index)
                    inference_data.x = aggregate_node_feature(inference_data.x, inference_data.edge_index)

        args.node_batch_size = int(
            training_data.x.size()[0] / (training_data.edge_index.size(1) / args.link_batch_size))

    #### Prepare the teacher and student model
    model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(args.device)

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1, args.num_layers,
                              args.dropout).to(args.device)
    pretrained_model = torch.load(
        "../saved-models/" + args.datasets + "-" + args.encoder + "_" + args.transductive + ".pkl")

    teacher_predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1, 2, args.dropout)
    teacher_predictor.load_state_dict(pretrained_model['predictor'], strict=True)
    teacher_predictor.to(args.device)

    t_h = torch.load("../saved-features/" + args.datasets + "-" + args.encoder + "_" + args.transductive + ".pkl")
    t_h = t_h['features'].to(args.device)

    for para in teacher_predictor.parameters():
        para.requires_grad = False

    evaluator = Evaluator(name='ogbl-ddi')

    n_classes = args.hidden_channels
    Discriminator_P = logits_pairs(n_classes, n_classes, n_classes, args.num_layers, args.dropout)
    Discriminator_P.to(args.device)

    if args.transductive == "transductive":
        if args.datasets != "collab":
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@20': Logger(args.runs, args),
                'Hits@30': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'AUC': Logger(args.runs, args),
            }
        elif args.datasets == "collab":
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'Hits@100': Logger(args.runs, args),
                'AUC': Logger(args.runs, args),
            }
    else:
        loggers = {
            'Hits@10': ProductionLogger(args.runs, args),
            'Hits@20': ProductionLogger(args.runs, args),
            'Hits@30': ProductionLogger(args.runs, args),
            'Hits@50': ProductionLogger(args.runs, args),
            'AUC': ProductionLogger(args.runs, args),
        }
    for run in range(args.runs):
        torch_geometric.seed.seed_everything(run + 1)

        model.reset_parameters()
        predictor.reset_parameters()
        Discriminator_P.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(predictor.parameters()), lr=args.lr)
        opt_D = torch.optim.Adam(Discriminator_P.parameters(), lr=args.discriminator_lr)

        cnt_wait = 0
        best_val = 0.0

        for epoch in range(1, 1 + args.epochs):
            if args.transductive == "transductive":
                loss = train(model, predictor, t_h, teacher_predictor, data, split_edge, optimizer, args, device,
                                 Discriminator_P, opt_D)
                results, h = test_transductive(model, predictor, data, split_edge, evaluator, args.link_batch_size,
                                               'mlp', args.datasets, args)
            else:
                loss = train(model, predictor, t_h, teacher_predictor, training_data, None, optimizer, args, device)
                results, h = test_production(model, predictor, val_data, inference_data, test_edge_bundle,
                                             negative_samples, evaluator, args.link_batch_size, 'mlp', args.datasets)
            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0
            else:
                cnt_wait += 1

            for key, result in results.items():
                loggers[key].add_result(run, result)
            if epoch % args.log_steps == 0:
                if args.transductive == "transductive":
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                else:
                    for key, result in results.items():
                        valid_hits, test_hits, old_old, old_new, new_new = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'valid: {100 * valid_hits:.2f}%, '
                              f'test: {100 * test_hits:.2f}%, '
                              f'old_old: {100 * old_old:.2f}%, '
                              f'old_new: {100 * old_new:.2f}%, '
                              f'new_new: {100 * new_new:.2f}%')
                print('---')
            if cnt_wait >= args.patience:
                break
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
    if args.transductive == "transductive":
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics()
            file.write(f'{key}:\n')
            best_results = []
            for r in loggers[key].results:
                r = 100 * torch.tensor(r)
                valid = r[:, 0].max().item()
                test1 = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test1))

            best_result = torch.tensor(best_results)
            r = best_result[:, 1]
            best = 0
            for i in r:
                if best < i:
                    best = i
            file.write(f'Test:{best:.4f}\n')
            file.write(f'Test: {r.mean():.4f} ± {r.std():.4f}\n')
    else:
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics()

            file.write(f'{key}:\n')
            best_results = []
            for r in loggers[key].results:
                r = 100 * torch.tensor(r)
                val = r[r[:, 0].argmax(), 0].item()
                test_r = r[r[:, 0].argmax(), 1].item()
                old_old = r[r[:, 0].argmax(), 2].item()
                old_new = r[r[:, 0].argmax(), 3].item()
                new_new = r[r[:, 0].argmax(), 4].item()
                best_results.append((val, test_r, old_old, old_new, new_new))

            best_result = torch.tensor(best_results)

            r = best_result[:, 0]
            file.write(f'  Final val: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            file.write(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            file.write(f'   Final old_old: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            file.write(f'   Final old_new: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            file.write(f'   Final new_new: {r.mean():.2f} ± {r.std():.2f}\n')
    file.close()
main()
