# -*- coding: utf-8 -*-
# fie:3


from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)

'''
We hereby focus on the bert version.
class AEN_GloVe(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AEN_GloVe, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()

        self.attn_k = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        context = self.squeeze_embedding(context, context_len)
        target = self.embed(target_indices)
        target = self.squeeze_embedding(target, target_len)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out
'''
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # print(adj.size(), hidden.size())
        output = torch.matmul(adj, hidden) / denom

        if self.bias is not None:
            return output + self.bias
        else:
            return output

def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

class BERT_GCN_PROMPT(nn.Module):
    def __init__(self, bert, opt, score_function='mlp', hidden_dim=None, out_dim=None, n_head=1):
        super(BERT_GCN_PROMPT, self).__init__()
        print(opt.bert_dim)
        if hidden_dim is None:
            hidden_dim = opt.bert_dim // n_head
        if out_dim is None:
            out_dim = opt.bert_dim

        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        self.gc1_1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc1_2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc1_3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc1_4 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc1_5 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc1_6 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc1_7 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc1_8 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc1_9 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_4 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_5 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_6 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_7 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_8 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2_9 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.score_function = score_function
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        # self.dropout = nn.Dropout(dropout)
        self.affine1 = nn.Parameter(torch.Tensor(768, 768))
        if self.score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(opt.bert_dim))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(opt.bert_dim, opt.bert_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.wte_enc = nn.Linear(5, 768)    #1

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x


    def forward(self, inputs):
        context, target, adj, adj_o, concat_segments_indices = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        bsz = context.shape[0]
        soft_prompt = torch.randint(5, 120, (bsz,1)).long().cuda()
        context=torch.concat((context, soft_prompt), dim=1)  # 1
        context=torch.concat((context, target), dim=1)   #1
        context_len = torch.sum(context != 0, dim=-1)  #1

        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context, return_dict=False)
        K1 = F.softmax(torch.bmm(torch.matmul(context, self.affine1), torch.transpose(context, 1, 2)), dim=-1)
        context = torch.bmm(K1, context)
        context = self.dropout(context)

        mb_size = context.shape[0]
        q_len = context.shape[1]
        adj = adj[:, :q_len, :q_len]
        adj_o = adj_o[:, :q_len, :q_len]
        graph_x = F.relu(self.gc1_1(context, adj))
        graph_x = F.relu(self.gc1_2(graph_x, adj))
        graph_x = F.relu(self.gc1_3(graph_x, adj))
        graph_x = F.relu(self.gc1_4(graph_x, adj))
        graph_x = F.relu(self.gc1_5(graph_x, adj))

        graph_x_o = F.relu(self.gc2_1(context, adj_o))
        graph_x_o = F.relu(self.gc2_2(graph_x_o, adj_o))
        graph_x_o = F.relu(self.gc2_3(graph_x_o, adj_o))
        graph_x_o = F.relu(self.gc2_4(graph_x_o, adj_o))
        graph_x_o = F.relu(self.gc2_5(graph_x_o, adj_o))

        _, _, bert_graph_k, _ = self.attn_k(context, context)
        _, _, bert_graph_q, _ = self.attn_q(graph_x, graph_x)

        kt = bert_graph_k.permute(0, 2, 1)
        qkt = torch.bmm(bert_graph_q, kt)
        score = torch.div(qkt, math.sqrt(self.opt.bert_dim))

        _, _, bert_graph_q_o, _ = self.attn_q(graph_x_o, graph_x_o)

        qkt_o = torch.bmm(bert_graph_q_o, kt)
        score_o = torch.div(qkt_o, math.sqrt(self.opt.bert_dim))

        score = F.softmax(score, dim=-1)
        score_o = F.softmax(score_o, dim=-1)

        output = torch.bmm(score, bert_graph_k)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)  #1
        output_o = torch.bmm(score_o, bert_graph_q_o)  # (n_head*?, q_len, hidden_dim)
        output_o = torch.cat(torch.split(output_o, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)  #1

        output = self.proj(output)  # (?, q_len, out_dim)
        output_o = self.proj(output_o)
        output = self.dropout(output)
        output_o = self.dropout(output_o)
        case_1 = F.softmax(torch.sum(output, dim=2))
        case_2 = F.softmax(torch.sum(output_o, dim=2))
        output = torch.div(torch.sum(output, dim=1), context_len.unsqueeze(1).float())
        output_o = torch.div(torch.sum(output_o, dim=1), context_len.unsqueeze(1).float())
        mutual_loss = js_div(output, output_o)
        output = 0.7*output + 0.3*output_o
        out = self.dense(output)
        return out, mutual_loss, case_2, case_1
