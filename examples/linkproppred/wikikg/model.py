#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataloader import TestDataset
from collections import defaultdict

from ogb.linkproppred import Evaluator


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, rel_init_scale, evaluator,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.pnorm = 1
        if model_name in ['TransE2', 'ConnE2']:
            self.pnorm = 2
        train_relations = rel_init_scale >= 0
        rel_init_scale = abs(rel_init_scale)
        train_entities = model_name!='TransEX'
        if model_name=='TransEX':
            model_name = 'TransE'
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        if model_name in ['HeadRE', 'TailRE']:
            self.relation_dim += 1
        if model_name in ['PairSE']:
            self.relation_dim += 2

        self.entity_embedding = nn.Parameter(
            torch.zeros(nentity, self.entity_dim),
            requires_grad=train_entities)
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim),
            requires_grad=train_relations)
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item()*rel_init_scale,
            b=self.embedding_range.item()*rel_init_scale,
        )

        if model_name in ['TuckER', 'Groups']:
            self.tensor_weights = nn.Parameter(
                torch.zeros(self.hidden_dim,self.hidden_dim,self.hidden_dim)) # head x tail x rel
            nn.init.uniform_(
                tensor=self.tensor_weights,
                a=-self.embedding_range.item()*rel_init_scale,
                b=self.embedding_range.item()*rel_init_scale,
            )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['BasE', 'TransE', 'Aligned', 'Aligned1', 'AlignedP', 'ConnE',
                              'TransE2', 'ConnE2', 'DistMult', 'ComplEx', 'RotatE', 'PairRE',
                              'PairSE', 'TransPro', 'HeadRE', 'TailRE', 'TuckER', 'Groups' ]:
            raise ValueError('model %s not supported' % model_name)

        if model_name in ['RotatE','Groups'] and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError(
                'ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name in ['PairRE','AlignedP','TransPro'] and not double_relation_embedding:
            raise ValueError('PairRE should use --double_relation_embedding')

        self.evaluator = evaluator

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(
                0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(
                0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == 'relations':
            head_part, tail_part = sample
#            print( head_part.size(), tail_part.size() )
            batch_size, negative_sample_size = head_part.size(0), self.nrelation

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 2]
            ).unsqueeze(1)

            relation = self.relation_embedding.view(1, negative_sample_size, -1)
#            print( head.size(), relation.size(), tail.size() )


        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'BasE': self.BasE,
            'TransE': self.TransE,
            'TransEX': self.TransE,
            'Aligned': self.Aligned,
            'Aligned1': self.Aligned1,
            'AlignedP': self.AlignedP,
            'ConnE': self.ConnE,
            'TransE2': self.TransE,
            'ConnE2': self.ConnE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'PairRE': self.PairRE,
            'PairSE': self.PairSE,
            'HeadRE': self.HeadRE,
            'TailRE': self.TailRE,
            'TransPro': self.TransPro,
#            'TuckER': self.TuckER,
            'Groups': self.Groups,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def BasE(self, head, relation, tail, mode):
        score = head - tail
        score = torch.norm(score, p=self.pnorm, dim=2)
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=self.pnorm, dim=2)
        return score

    def Aligned(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = relation.expand_as(tail)
        else:
            score = relation.expand_as(head)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        return cos( score, (tail - head) )

    def Aligned1(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        return torch.norm(score, p=self.pnorm, dim=2)-torch.norm(head-tail, p=self.pnorm, dim=2)

    def ConnE(self, head, relation, tail, mode):
        score = head - tail
        relnorm = torch.norm(relation, p=2, dim=2)
        normrel = torch.div(relation, torch.sqrt(
            relnorm + torch.ones_like(relnorm)))
#        print( score.size(), relation.size(), torch.einsum( 'mpi,mni->mn', relation, score ).size() )
        score = self.gamma.item() - torch.norm(score, p=self.pnorm, dim=2) - \
            torch.einsum('mpi,mni->mn', normrel, score)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def PairRE(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def PairSE(self, head, relation, tail, mode):
        hidden_dim = head.size(2)
        projections, scales = torch.split(relation, [2*hidden_dim,2], dim=2)
        re_head, re_tail = torch.chunk(projections.detach(), 2, dim=2)
        scale_head, scale_tail = torch.chunk(scales, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * scale_head * re_head - tail * scale_tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def HeadRE(self, head, relation, tail, mode):
        hidden_dim = head.size(2)
        re_head, scale_tail = torch.split(relation, [hidden_dim,1], dim=2)
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        score = head * re_head - (tail * scale_tail)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TailRE(self, head, relation, tail, mode):
        hidden_dim = tail.size(2)
        re_tail, scale_head = torch.split(relation, [hidden_dim,1], dim=2)
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        score = (head * scale_head) - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def AlignedP(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        if mode == 'head-batch':
            score = torch.norm(head * re_head, p=1, dim=2)
        else:
            score = torch.norm(tail * re_tail, p=1, dim=2)
        return score

    def TransPro(self, head, relation, tail, mode):
        relation, projection = torch.chunk(relation, 2, dim=2)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(projection * score, p=self.pnorm, dim=2)
        return score

    def Groups(self, head, relation, tail, mode):
        head_h, head_t = torch.chunk(head, 2, dim=2)
        tail_h, tail_t = torch.chunk(tail, 2, dim=2)
        WR = torch.matmul( self.tensor_weights, relation )
        print( WR.size() )
        print( torch.matmul( WR, tail_t ).size() )
        
        if mode == 'head-batch':
            score = torch.matmul( head_h, torch.matmul( WR, tail_t ) )
        else:
            score = torch.matmul( torch.matmul( head_h, WR ), tail_t )
            
        print( score.size() )
        return self.gamma.item() - score.sum(dim=2)

    def print_relation_embedding(self, filename, args):
        dump = open(filename,"w")
        if args.print_relation_option=='list':
            rel = self.relation_embedding.to(torch.device("cpu")).detach()
            for r in rel:
                print( r.numpy(), file=dump )
        elif args.print_relation_option=='triple-add':
            g = args.gamma
            rel = self.relation_embedding.detach()
            for i in range(len(rel)-1):
                for j in range(i+1,len(rel)):
                    v = rel[i] + rel[j]
                    rel_v = rel - v
                    score = torch.norm(rel_v, p=self.pnorm, dim=1).to(torch.device("cpu")).detach()
                    if i==0 and j==1:
                        print( rel_v.size(), score.size() )
                    for k in range(len(score)):
                        if k!=i and k!=j and score[k]<g:
                            print( i, j, k, score[k].item(), file=dump )
                            
                    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(
            train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - \
                (subsampling_weight * positive_score).sum() / \
                subsampling_weight.sum()
            negative_sample_loss = - \
                (subsampling_weight * negative_score).sum() / \
                subsampling_weight.sum()

#        if args.contact_loss:
#            c_loss = F.logsigmoid(self.gamma.item() - negative_score).sum(dim=1)
#            loss = (positive_sample_loss + negative_sample_loss)/2 + args.contact_alpha*c_loss
#        else:
        loss = (positive_sample_loss + negative_sample_loss)/2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3)**3 +
                model.relation_embedding.norm(p=3).norm(p=3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, args, random_sampling=False, dump_all=False):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if dump_all or args.dump_sample>=0:
            dump = open(args.dump_filename, "w")
            min_val = args.hist_minval
            range_val = args.hist_maxval - min_val
            if args.test_dump_hist > 0:
                break_list = np.zeros(args.test_dump_hist, dtype=float)
                for n in range(0, args.test_dump_hist):
                    break_list[n] = min_val + n*range_val/args.test_dump_hist
                print("# brks<-c(", end='', file=dump)
                print("{:.2f}".format(break_list[0]), end='', file=dump)
                for n in range(1, args.test_dump_hist):
                    print(",{:.2f}".format(break_list[n]), end='', file=dump)
                print(")", file=dump)

        # Prepare dataloader for evaluation

        if args.swap_relations:
            test_dataloader_rel = DataLoader(
                TestDataset(
                    test_triples,
                    args,
                    'relations',
                    False
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TestDataset.collate_fn
            )
            test_dataset_list = [test_dataloader_rel]
        else:
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    args,
                    'head-batch',
                    random_sampling
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TestDataset.collate_fn
            )
            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    args,
                    'tail-batch',
                    random_sampling
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TestDataset.collate_fn
            )
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        test_logs = defaultdict(list)

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        if args.test_dump_byrel:
            hist_byrel = np.zeros((2,args.nrelation,args.test_dump_hist), dtype=int)

        if args.test_log_steps<0:
            step = 1
            args.test_log_steps = total_steps
            
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.dump_sample>=0 and args.dump_sample<len(positive_sample):
                        (ps,ns) = (positive_sample, negative_sample)
                        
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), mode)

                    batch_results = model.evaluator.eval({'y_pred_pos': score[:, 0],
                                                          'y_pred_neg': score[:, 1:]})
                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if args.dump_sample>=0 and args.dump_sample<len(positive_sample):
                        j = args.dump_sample
#                        h = positive_sample[j,0].item()
#                        r = positive_sample[j,1].item()
#                        t = positive_sample[j,2].item()
                        s = score[j].to(torch.device("cpu"))
                        print( step, mode, ps[j].numpy(), ' ', ns[j].numpy(), ' ', s.numpy(), file=dump )
                        
                    if dump_all:
                        rels = positive_sample[:,1].to(torch.device("cpu"))
                        score2 = score.to(torch.device("cpu"))
                        for j in range(len(score2)):
                            s = score2[j]
                            rel = rels[j].item()
#                            print( 'step', step, 'item', j, 'relation', rel, file=dump)
                            if not args.test_dump_byrel:
                                hist = np.zeros(args.test_dump_hist, dtype=int)
                                print('item(', step, ",", s[0].item(), file=dump)
                            for i in range(0 if args.test_dump_byrel else 1, len(s)):
                                if args.test_dump_hist > 0:
                                    n = int(args.test_dump_hist *
                                            (s[i].item()-min_val)/range_val)
                                    if n < 0:
                                        print('# score', s[i].item(), 'less than', min_val, file=dump)
                                        n = 0
                                    if n >= args.test_dump_hist:
                                        print('# score', s[i].item(), 'greater than',
                                              range_val+min_val, file=dump)
                                        n = args.test_dump_hist-1
                                    if args.test_dump_byrel:
                                        hist_byrel[0 if i==0 else 1][rel][n] += 1
                                    else:
                                        hist[n] += 1
                                        if args.test_dump_hist == 0:
                                            print(',', s[i].item(), file=dump)
                            if not args.test_dump_byrel:
                                if args.test_dump_hist > 0:
                                    print(", c(", hist[0], end='', file=dump)
                                    for n in range(1, args.test_dump_hist):
                                        print(",", hist[n], end='', file=dump)
                                    print("))\n", file=dump)
                                else:
                                    print(")\n", file=dump)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' %
                                     (step, total_steps))
                        if step>0 and args.test_dump_byrel:
                            print('step neg relation mean sd hist', file=dump)
                            for i in range(2):
                                for j in range(args.nrelation):
                                    num = hist_byrel[i][j].sum()
                                    if num>=1:
                                        vals = break_list * hist_byrel[i][j]
                                        valsq = break_list * vals
                                        m =  vals.sum() / num
                                        sd = valsq.sum() / num - m*m
                                        print(step, i, j, m, sd, hist_byrel[i][j].tolist(), file=dump)

                    step += 1


            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics
