import os
from os.path import join
import argparse
import numpy as np
import torch

from utils.noter import Noter
from utils.constant import MAPPING_RAW, IDX_PAD


def main():
    parser = argparse.ArgumentParser('CPMR (CIKM\'23)')
    parser.add_argument('--data', type=str, default='garden', help='garden, video, game, ml')
    parser.add_argument('--ver', type=str, default='1.0', help='version name')

    # MTL
    parser.add_argument('--no_pmtl', action='store_true', help='ablation, no PMTL module')
    parser.add_argument('--no_ctx', action='store_true', help='ablation, remove context branch')
    parser.add_argument('--no_his', action='store_true', help='ablation, remove history branch')

    # context-aware cgnn
    parser.add_argument('--len_ctx', type=float, default=5, help='length of context window (in days)')
    parser.add_argument('--alpha_spectrum', type=float, default=0.98, help='limited spectrum of graph')
    parser.add_argument('--k_inv', type=int, default=10, help='order of Neumann series')
    parser.add_argument('--k_exp', type=int, default=3, help='order of Chebyshev polynomial')

    # hyperparameters
    parser.add_argument('--d_latent', type=int, default=128)
    parser.add_argument('--n_tbptt', type=int, default=20, help='truncated back propagation through time')
    parser.add_argument('--n_neg_sample', type=int, default=8)
    parser.add_argument('--no_bn', action='store_true', help='deactivate batch norm')

    # optimizer
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l2', type=float, default=5e-3, help='weight_decay')
    parser.add_argument('--lr_step', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.2, help='i.e. gamma value')
    parser.add_argument('--n_lr_decay', type=int, default=5)

    # training settings
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--k_metric', type=int, default=10)
    parser.add_argument('--proportion_train', type=float, default=0.8, help='proportion of training set')
    parser.add_argument('--n_batch_load', type=int, default=10, help='number of thread loading batches')
    parser.add_argument('--es_patience', type=int, default=10)

    args = parser.parse_args()

    if args.no_ctx or args.no_his:
        if args.no_ctx and args.no_his:
            raise NotImplementedError('Cannot specify no_ctx and no_his together.')
        from trainer_no_ctx_his import Trainer
    else:
        from trainer import Trainer

    args.idx_pad = IDX_PAD
    (args.dataset, args.f_raw) = MAPPING_RAW[args.data]
    args.lr_min = args.lr ** (args.n_lr_decay + 1)
    args.device = torch.device('cuda:' + str(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')

    args.f_raw = join('../CPMR/data', 'raw', args.f_raw)
    args.f_csv = join('../CPMR/data', 'core_5', args.dataset + '_5.csv')
    args.path_log = join('log')
    for p in (args.path_log):
        if not os.path.exists(p):
            os.makedirs(p)

    # initialize
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    noter = Noter(args)
    trainer = Trainer(args, noter)

    # modeling
    recall_best, mrr_best, loss_best = 0, 0, 1e5
    res_recall_final, res_mrr_final, res_loss_final = [0]*5, [0]*5, [0]*5
    epoch, es_counter = 0, 0
    lr_register = args.lr

    for epoch in range(1, args.n_epoch+1):
        noter.log_msg(f'\n[Epoch {epoch}]')
        recall_val, mrr_val, loss_rec_val = trainer.run_one_epoch()
        trainer.scheduler.step()

        # models selection
        msg_best_val = ''
        if loss_rec_val < loss_best:
            loss_best = loss_rec_val
            msg_best_val += f' loss |'

        if recall_val > recall_best:
            recall_best = recall_val
            msg_best_val += f' recall |'

        if mrr_val > mrr_best:
            mrr_best = mrr_val
            msg_best_val += f' mrr |'

        if len(msg_best_val) > 0:
            res_test = trainer.run_test()
            noter.log_msg('\t| new   |' + msg_best_val)

            if 'loss' in msg_best_val:
                res_loss_final = [epoch] + res_test
            if 'recall' in msg_best_val:
                res_recall_final = [epoch] + res_test
            if 'mrr' in msg_best_val:
                res_mrr_final = [epoch] + res_test

        # lr
        lr_current = trainer.scheduler.get_last_lr()[0]
        if lr_register != lr_current:
            if trainer.optimizer.param_groups[0]['lr'] == args.lr_min:
                noter.log_msg(f'\t| lr    | reaches btm | {args.lr_min:.2e} |')
            else:
                noter.log_msg(f'\t| lr    | from {lr_register:.2e} | to {lr_current:.2e} |')
                lr_register = lr_current

        # early stop
        if loss_rec_val > loss_best:
            es_counter += 1
            noter.log_msg(f'\t| es    | {es_counter} / {args.es_patience} |')
        elif es_counter != 0:
            es_counter = 0
            noter.log_msg(f'\t| es    | 0 / {args.es_patience} |')

        if es_counter >= args.es_patience:
            break

    noter.log_final_result(epoch, {
        'loss  ': res_loss_final,
        'recall': res_recall_final,
        'mrr   ': res_mrr_final,
    })


if __name__ == '__main__':
    main()
