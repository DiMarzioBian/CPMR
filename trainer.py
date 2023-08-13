import time
import torch
from tqdm import tqdm

from dataloaders.dataloader import get_dataloader
from utils.misc import cal_mrr, cal_recall


class Trainer(object):
    def __init__(self, args, noter):
        # experiment
        if not args.no_pmtl:
            from models.CPMR import CPMR
        else:
            from models.CPMR_no_pmtl import CPMR

        self.trainloader, self.valloader, self.testloader = get_dataloader(args, noter)
        self.model = CPMR(args).to(args.device)
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=args.lr,
                                          weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        self.noter = noter

        self.len_train_dl = len(self.trainloader)
        self.len_val_dl = len(self.valloader)
        self.len_test_dl = len(self.testloader)

        self.n_tbptt = args.n_tbptt
        self.k_metric = args.k_metric

        self.xu_t_plus_his = None
        self.xi_t_plus_his = None
        self.xu_t_plus_ctx = None
        self.xi_t_plus_ctx = None

        self.noter.log_brief()

    def detach_states(self):
        self.xu_t_plus_his = self.xu_t_plus_his.detach()
        self.xi_t_plus_his = self.xi_t_plus_his.detach()
        self.xu_t_plus_ctx = self.xu_t_plus_ctx.detach()
        self.xi_t_plus_ctx = self.xi_t_plus_ctx.detach()

    def run_one_epoch(self):
        self.xu_t_plus_his, self.xi_t_plus_his, self.xu_t_plus_ctx, self.xi_t_plus_ctx = self.model.get_init_states()
        self.run_train()
        return self.run_valid()

    def run_train(self):
        loss_tbptt, loss_total, count_tbptt, count_total = 0., 0., 0, 0
        time_start = time.time()

        self.model.train()
        self.optimizer.zero_grad()

        for batch in tqdm(self.trainloader, desc='  - training', leave=False):
            loss_batch, _, _, self.xu_t_plus_his, self.xi_t_plus_his, self.xu_t_plus_ctx, self.xi_t_plus_ctx = \
                self.model(batch, self.xu_t_plus_his, self.xi_t_plus_his, self.xu_t_plus_ctx, self.xi_t_plus_ctx)

            loss_tbptt += loss_batch
            loss_total += loss_batch.item() / self.len_train_dl
            count_tbptt += 1
            count_total += 1

            if (count_tbptt % self.n_tbptt) == 0 or count_total == self.len_train_dl:
                loss = loss_tbptt / count_tbptt
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                count_tbptt, loss_tbptt = 0, 0
                self.detach_states()

        self.noter.log_train(loss_total, time.time() - time_start)

    def run_valid(self):
        # validating phase
        rank_val, loss_val = self.rollout('validating')

        mrr_val = cal_mrr(rank_val)
        recall_val = cal_recall(rank_val, self.k_metric)

        self.noter.log_valid(loss_val, mrr_val, recall_val)
        return recall_val, mrr_val, loss_val

    def run_test(self):
        # testing phase
        rank_test, *_ = self.rollout('testing')

        mrr_test = cal_mrr(rank_test)
        recall_test = cal_recall(rank_test, self.k_metric)

        self.noter.log_test(mrr_test, recall_test)
        return [mrr_test, recall_test]

    def rollout(self, mode: str):
        """ rollout evaluation """
        self.model.eval()
        loss_epoch = 0

        if mode == 'validating':
            dl = self.valloader
            len_dl = self.len_val_dl
        else:
            assert mode == 'testing'
            dl = self.testloader
            len_dl = self.len_test_dl

        if self.xu_t_plus_his is None:
            self.xu_t_plus_his, self.xi_t_plus_his, self.xu_t_plus_ctx, self.xi_t_plus_ctx = \
                self.model.get_init_states()

        rank_u = []
        with torch.no_grad():
            for batch in tqdm(dl, desc='  - ' + mode, leave=False):
                (tgt_u, tgt_i) = batch[5:7]
                loss_batch, zu_pos, zi, self.xu_t_plus_his, self.xi_t_plus_his, self.xu_t_plus_ctx, self.xi_t_plus_ctx \
                    = self.model(batch, self.xu_t_plus_his, self.xi_t_plus_his, self.xu_t_plus_ctx, self.xi_t_plus_ctx)

                loss_epoch += loss_batch.item() / len_dl
                rank_u_batch = self.cal_rank(zu_pos, zi[1:], tgt_i - 1)
                rank_u.extend(rank_u_batch)

        return rank_u, loss_epoch

    def cal_rank(self, zu, zi, tgt_i):
        scores = self.model.predictor(zu, zi.squeeze(1).unsqueeze(0))
        rank_u = []
        for line, i in zip(scores, tgt_i):
            r = (line >= line[i]).sum().item()
            rank_u.append(r)
        return rank_u
