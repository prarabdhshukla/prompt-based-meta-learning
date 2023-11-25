
import torch
import json
from transformers import DebertaTokenizer, DebertaModel

bert =  DebertaModel.from_pretrained("microsoft/deberta-base")
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
emb_M = bert.embeddings.word_embeddings.weight

# %%
emb_M.shape

# %%
def show_tokenized_name(idx2name):
    for key in idx2name:
        name_list = idx2name[key]
        for name in name_list:
             print(tokenizer.tokenize(name))

def save_ebd_list(idx2name,outdir):
    cl2ebd_list = dict()
    for cl in idx2name:
        vec_list = []
        names = idx2name[cl]
        for name in names:
            tokenized_name = tokenizer.tokenize(name)
            indexed_name = tokenizer.convert_tokens_to_ids(tokenized_name)
            ebd = torch.mean(emb_M[indexed_name],0).tolist()
            vec_list.append(ebd)
        cl2ebd_list[cl] = vec_list

    with open(outdir,'w') as fout:
        fout.write(json.dumps(cl2ebd_list))
        fout.write('\n')

# for benchmark in {"Amazon","Reuters","HuffPost"}:
#     idx2name = json.load(open(f'data/{benchmark}/candidate_words.json','r'))
#     save_ebd_list(idx2name,f'data/{benchmark}/candidate_ebds.json')


# candidate ebds for FewRel (from P-info)

word2vec={} # word： vec
WikiData={} # Pid: dictionary name alias etc.
num2embed={}

def read_info(file_name):
    WikiDatafile=json.load(open(file_name,'r',encoding='utf-8'))
    for relation in WikiDatafile:
        name_list = []
        name_list.append(relation['name'].lower())
        for i in range(len(relation['alias'])):
            name_list.append(relation['alias'][i].lower())
        relation['name'] =name_list
        WikiData[relation['id']]= name_list

read_info('data/fewrel/P_info.json')
cl2ebd_list = dict()

for cl in WikiData:
    vec_list = []
    for name in WikiData[cl]:
        name = name.lower()
        name = name.split()
        tokenized_name = []
        for w in name:
            tokenized_name += tokenizer.tokenize(w)
        indexed_name = tokenizer.convert_tokens_to_ids(tokenized_name)
        ebd = torch.mean(emb_M[indexed_name],0).tolist()

        vec_list.append(ebd)
    cl2ebd_list[cl] = vec_list

with open('data/fewrel/candidate_ebds.json','w') as fout:
    fout.write(json.dumps(cl2ebd_list))
    fout.write('\n')

# %%
import json
from collections import defaultdict
import numpy as np
import time

# %% [markdown]
# #### Training the dataset

# %%
import os
import math
import json
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
from torch.optim import AdamW

# %%
class FewshotDataset(data.Dataset):
    def __init__(self,file_name,N,K,Q,noise_rate):
        super(FewshotDataset,self).__init__()
        if not os.path.isfile(file_name):
            raise Exception("[ERROR] Data file doesn't exist")

        self.json_data = json.load(open(file_name,'r',encoding='utf-8'))
        self.classes = list(self.json_data.keys())
        self.N, self.K, self.Q = N,K,Q
        self.noise_rate = noise_rate

    def __len__(self):
        return 1000000000

    def __getitem__(self,index):
        N, K, Q = self.N, self.K, self.Q
        class_name = random.sample(self.classes,N) # N categories
        support, support_label, query, query_label = [],[],[],[]
        for i,name in enumerate(class_name):
            cl = self.json_data[name]
            samples = random.sample(cl,K+Q)
            for j in range(K):
                support.append([samples[j],i])
            for j in range(K,K+Q):
                query.append([samples[j],i])

        query=random.sample(query,N*Q) # shuffle query order

        for i in range(N*K):
            support_label.append(support[i][1])
            support[i]=support[i][0]

        for i in range(N*Q):
            query_label.append(query[i][1])
            query[i]=query[i][0]

        if self.noise_rate>0: # replace support instance with noised instance from other categories
            other_classes=[]
            for _ in self.classes:
                if _ not in class_name:
                    other_classes.append(_)
            for i in range(N*K):
                if(random.randint(1,10)<=self.noise_rate):
                    noise_name=random.sample(other_classes,1)
                    cl=self.json_data[noise_name[0]]
                    support[i]=random.sample(cl,1)[0]

        support_label = torch.tensor(support_label).long()
        query_label = torch.tensor(query_label).long()

        if torch.cuda.is_available():support_label,query_label=support_label.cuda(),query_label.cuda()
        return class_name,support,support_label,query,query_label

# %%
def gelu(x):
    return x  * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# %%
class Initializer(nn.Module):
    def __init__(self, N, K, data_dir):
        super(Initializer,self).__init__()
        self.n_way = N
        self.k_shot = K
        self.embedding_dim = 768

        candidate_info = data_dir['candidates'] # candidate word info
        if candidate_info is None or not os.path.isfile(candidate_info):
            raise Exception("[ERROR] candidate words information file doesn't exist")

        self.cl2embed = json.load(open(candidate_info,'r')) # {class_name: candidate word embeddings}

        for key in self.cl2embed.keys():
            self.cl2embed[key] = torch.Tensor(self.cl2embed[key]).cuda()

    def get_embedding(self, class_names):
        # read candidate word embeddings from the class name
        res = []
        for i in range(len(class_names)):
            class_name = class_names[i]
            vec_list = self.cl2embed[class_name].float()
            res.append(vec_list)
        return res  # [N * [candidate word embeddings]]

    def forward(self, inputs): # inputs: [N * [candidate word embeddings]]
        # average pooling
        W = torch.zeros(len(inputs), self.embedding_dim).cuda()
        for idx in range (len(inputs)):
            W[idx] = torch.mean(inputs[idx], 0).requires_grad_(True) # [hidden_size] candidates mean pooler
            # W[idx] = inputs[idx][0].requires_grad_(True) # [hidden_size] without kg
        if self.k_shot == 1:
            W = F.normalize(W,dim=-1)
        elif self.k_shot == 5:
            W = 0.5 * F.normalize(W,dim=-1)

        return W

# %%
from transformers import DebertaTokenizer, DebertaModel
from transformers import DebertaForMaskedLM

class BERT(nn.Module):
    def __init__(self, N, max_length, data_dir, blank_padding=True):
        super(BERT,self).__init__()
        self.cuda = torch.cuda.is_available()
        self.n_way = N
        self.max_length = max_length
        self.blank_padding = blank_padding
        # self.pretrained_path = 'bert-base-uncased'

        # bert_model = BertModel.from_pretrained(self.pretrained_path)
        bert_model = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        self.bert_ebd = bert_model.embeddings
        self.bert_encoder = bert_model.encoder

        # self.tokenizer = BertTokenizer.from_pretrained()
        self.tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
        self.dropout = nn.Dropout(data_dir['pb_dropout'])
        self.benchmark = data_dir['benchmark']

        mlm = DebertaForMaskedLM.from_pretrained("microsoft/deberta-base")
        D = mlm.cls.state_dict()
        (pred_bias, tf_dw, tf_db, tf_lnw, tf_lnb, dec_w, dec_b) = (D['predictions.bias'],
                                                            D['predictions.transform.dense.weight'],
                                                            D['predictions.transform.dense.bias'],
                                                            D['predictions.transform.LayerNorm.weight'],
                                                            D['predictions.transform.LayerNorm.bias'],
                                                            D['predictions.decoder.weight'],
                                                            D['predictions.decoder.bias'])
        self.LayerNorm = nn.LayerNorm(768,eps = 1e-12)
        self.LayerNorm.weight.data, self.LayerNorm.bias.data = tf_lnw,tf_lnb
        self.tf_dense = nn.Linear(768,768)
        self.tf_dense.weight.data,self.tf_dense.bias.data = tf_dw,tf_db

        # soft template params
        self.soft_prompt = nn.Parameter(torch.rand(4,768))
        soft_token = ['is', '[MASK]', 'of', '.']

        soft_token_id = self.tokenizer.convert_tokens_to_ids(soft_token)
        for i in range(len(soft_token)):
            self.soft_prompt.data[i] = self.bert_ebd.word_embeddings.weight.data[soft_token_id[i]]

    def forward(self,inputs):
        return self.forward_FewRel(inputs)

    def forward_FewRel(self,inputs): # [raw_tokens_dict * (N*K or total_Q)]
        input_ebds, MASK_INDs,att_masks,outputs = [],[],[],[]
        for _ in inputs:
            indexed_token, indexed_head, indexed_tail, avai_len = self.tokenize_FewRel(_)
            after_ebd_text = self.bert_ebd.word_embeddings(indexed_token) # [1,avai_len] ——> [1, avai_len, 768]
            after_ebd_head = self.bert_ebd.word_embeddings(indexed_head)  # [1,len_head] ——> [1, len_head, 768]
            after_ebd_tail = self.bert_ebd.word_embeddings(indexed_tail)  # [1,len_tail] ——> [1, len_tail, 768]
            input_ebd = torch.cat((after_ebd_text, after_ebd_head, self.soft_prompt[:3].unsqueeze(0)),1) # text head is [mask] of

            MASK_INDs.append(avai_len + indexed_head.shape[-1] + 1)
            input_ebd = torch.cat((input_ebd, after_ebd_tail, self.soft_prompt[3].unsqueeze(0).unsqueeze(0), self.bert_ebd.word_embeddings(torch.tensor(102).cuda()).unsqueeze(0).unsqueeze(0) ),1) # text head is [mask] of tail . [SEP]

            # mask calculation
            att_mask = torch.zeros(1,self.max_length)
            if self.cuda: att_mask = att_mask.cuda()
            att_mask[0][:input_ebd.shape[1]]=1 # [1, max_length]

            # padding tensor
            while input_ebd.shape[1] < self. max_length:
                input_ebd = torch.cat((input_ebd, self.bert_ebd.word_embeddings(torch.tensor(0).cuda()).unsqueeze(0).unsqueeze(0)), 1)

            input_ebd = input_ebd[:,:self.max_length,:]
            input_ebds.append(input_ebd)

            input_shape = att_mask.size()
            device = indexed_token.device

            extented_att_mask = self.get_extended_attention_mask(att_mask, input_shape,device)
            att_masks.append(extented_att_mask)

        input_ebds = torch.cat(input_ebds,0) # [N*K, max_length，768]
        tensor_masks = torch.cat(att_masks,0) # [N*K, max_length] then extend
        sequence_output= self.bert_encoder(self.bert_ebd(inputs_embeds = input_ebds) , attention_mask = tensor_masks).last_hidden_state # [N*K, max_length, bert_size]


        for i in range(input_ebds.size(0)):
            outputs.append(self.entity_start_state(MASK_INDs[i],sequence_output[i]))
            # [[1,bert_size*2] * (N*K)]
        tensor_outputs = torch.cat(outputs,0)  # [N*K,bert_size*2=hidden_size]

        # dropout
        tensor_outputs = self.dropout(tensor_outputs)

        return tensor_outputs

    def entity_start_state(self,MASK_IND,sequence_output): #  sequence_output: [max_length, bert_size]
        if MASK_IND >= self.max_length:
            MASK_IND = 0
        res = sequence_output[MASK_IND]
        res = self.LayerNorm(gelu(self.tf_dense(res)))

        return res.unsqueeze(0) # [1, hidden_size]

    def tokenize_FewRel(self,inputs): #input: raw_tokens_dict
        tokens = inputs['tokens']
        pos_head = inputs['h'][2][0]
        pos_tail = inputs['t'][2][0]

        re_tokens,cur_pos = ['[CLS]',],0

        for token in tokens:
            token=token.lower()
            if cur_pos == pos_head[0]:
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0]:
                re_tokens.append('[unused1]')

            re_tokens+=self.tokenizer.tokenize(token)

            if cur_pos==pos_head[-1]-1: re_tokens.append('[unused2]')
            if cur_pos==pos_tail[-1]-1: re_tokens.append('[unused3]')

            cur_pos+=1
        re_tokens.append('[SEP]')

        head = []
        tail = []
        for cur_pos in range(pos_head[0],pos_head[-1]):
            head += self.tokenizer.tokenize(tokens[cur_pos])
        for cur_pos in range(pos_tail[0],pos_tail[-1]):
            tail += self.tokenizer.tokenize(tokens[cur_pos])

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        indexed_head = self.tokenizer.convert_tokens_to_ids(head)
        indexed_tail = self.tokenizer.convert_tokens_to_ids(tail)
        avai_len = len(indexed_tokens)

        indexed_tokens = torch.tensor(indexed_tokens).unsqueeze(0).long()
        indexed_head = torch.tensor(indexed_head).unsqueeze(0).long()
        indexed_tail = torch.tensor(indexed_tail).unsqueeze(0).long()

        if self.cuda: indexed_tokens,indexed_head,indexed_tail = indexed_tokens.cuda(), indexed_head.cuda(), indexed_tail.cuda()
        return indexed_tokens, indexed_head, indexed_tail, avai_len


# %%
class PBML(nn.Module):
    def __init__(self, B, N, K, max_length, data_dir):
        nn.Module.__init__(self)

        self.batch = B
        self.n_way = N
        self.k_shot = K
        self.max_length = max_length
        self.data_dir = data_dir
        self.hidden_size = 768 # bert-base

        self.cost = nn.NLLLoss()
        self.coder = BERT(N,max_length,data_dir)
        self.initializer = Initializer(N,K, data_dir)

        self.W = [None] * self.batch # label word embedding matrix

    def loss(self,logits,label):
        return self.cost(logits.log(),label.view(-1))

    def accuracy(self,logits,label):
        label = label.view(-1)
        _, pred = torch.max(logits,1)
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def forward(self, inputs, W): # inputs: [N*K or total_Q, hidden_size]
                                  # W: [n_way, hidden_size]
        # s2w
        logits_for_instances = F.linear(inputs, W, bias=None) #[N*K or total_Q or 1 ,n_way]
        # w2s
        logits_for_classes = F.linear(W, torch.mean(inputs.view(self.n_way, inputs.shape[0]//self.n_way,768),dim=1), bias=None)

        return F.softmax(logits_for_instances,dim=-1), F.softmax(logits_for_classes,dim=-1)

    def get_info(self,class_names): # list of class_name
        return self.initializer.get_embedding(class_names) # [N * [candidate word embeddings]]

    def prework(self,candidate_word_embeddings): # meta-info: [N, hidden_size]
                                                # support:   [N*K, bert_size]
        return self.initializer(candidate_word_embeddings)

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# %%
# -*- coding: utf-8 -*-
import torch
from torch import autograd
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup
import sys


def fast_tuning(W,support,support_label,query,net,steps,task_lr,N,K):
    '''
       W:               label word embedding matrix                             [N, hidden_size]
       support:         support instance hidden states at [MASK] place          [N*K, hidden_size]
       support_label:   support instance label id:                              [N*K]
       query:           query instance hidden states at [MASK] place            [total_Q, hidden_size]
       steps：          fast-tuning steps
       task_lr:         fast-tuning learning rate for task-adaptation
    '''
    prototype_label = torch.tensor( [i for i in range(N)]).cuda() # [0,1,2,...N]
    # attention score calc
    idx = torch.zeros(N*K).long().cuda()
    for i in range(N): idx[i*K:(i+1)*K] = i # [0,0,...0,1,1...1,...N-1...N-1]
    att=(support * W[idx]).sum(-1).reshape(N,K) # ([N*K,bert_size]·[N*K,bert_size]).sum(-1) = [N*K] ——>  [N,K]
    T = 3
    att = F.softmax(att/T,-1).detach() # [N,K]
    # att: attention scores α_i^j

    for _ in range(steps):
        logits_for_instances, logits_for_classes = net(support,W) # [N*K, N], [N, N]
        if att is None:
            loss_s2v = net.loss(logits_for_instances, support_label)
            loss_v2s = net.loss(logits_for_classes, prototype_label)

            loss = loss_s2v + loss_v2s

            grads = autograd.grad(loss,W)
            W = W - task_lr*grads[0]
        else:
            Att = att.flatten() # [N*K]
            loss = torch.FloatTensor([0.0] * (N*K)).cuda()
            for i in range(N*K):
                loss[i]  = net.loss(logits_for_instances[i].unsqueeze(0),support_label[i])/N
            loss_tot = Att.dot(loss)
            grad = autograd.grad(loss_tot,W)
            W = W - task_lr*grad[0]

    logits_q = net(query, W)[0] # [total_Q, n_way]
    return logits_q

def train_one_batch(idx,class_names,support0,support_label,query0,query_label,net,steps,task_lr):
    '''
    idx:                batch index
    class_names：       N categories names (or name id)             List[class_name * N]
    support0:           raw support texts                           List[{tokens:[],h:[],t:[]} * (N*K)]
    support_label:      support instance labels                     [N*K]
    query0:             raw query texts                             List[{tokens:[],h:[],t:[]} * total_Q]
    query_label:        query instance labels                       [total_Q]
    net:                PBML model
    steps：             fast-tuning steps
    task_lr:            fast-tuning learning rate for task-adaptation
    '''
    N, K = net.n_way, net.k_shot

    support, query = net.coder(support0), net.coder(query0) # [N*K,bert_size]

    candidate_word_embeddings =net.get_info(class_names) # [N * [candidate word embeddings]]

    net.W[idx] = net.prework(candidate_word_embeddings)

    logits_q = fast_tuning(net.W[idx],support,support_label,query,net,steps,task_lr,N,K)

    return net.loss(logits_q, query_label),   net.accuracy(logits_q, query_label)


def test_model(data_loader,model,val_iter,steps,task_lr):
    accs=0.0
    model.eval()

    for it in range(val_iter):
        net = model
        class_name,support,support_label,query,query_label = data_loader[0]
        loss,right = train_one_batch(0,class_name, support, support_label,query,query_label,net,steps,task_lr)
        accs += right
        sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')
        sys.stdout.flush()

    return accs/val_iter


def train_model(model:PBML, B,N,K,Q,data_dir,
            meta_lr=5e-5,
            task_lr=1e-2,
            weight_decay = 1e-2,
            train_iter=2000,
            val_iter=2000,
            val_step=50,
            steps=30,
            save_ckpt = None,
            load_ckpt = None,
            best_acc = 0.0,
            fp16 = False,
            warmup_step = 200):

    n_way_k_shot = str(N) + '-way-' + str(K) + '-shot'
    print('Start training ' + n_way_k_shot)
    cuda = torch.cuda.is_available()
    if cuda: model = model.cuda()

    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('ignore {}'.format(name))
                continue
            print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)

    data_loader={}
    data_loader['train'] = FewshotDataset(data_dir['train'],N,K,Q,data_dir['noise_rate'])
    data_loader['val'] = FewshotDataset(data_dir['val'],N,K,Q,data_dir['noise_rate'])
    # data_loader['test'] = FewshotDataset(data_dir['test'],N,K,Q,data_dir['noise_rate'])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    coder_named_params = list(model.coder.named_parameters())

    for name, param in coder_named_params:
        if name in {'bert_ebd.word_embeddings.weight','bert_ebd.position_embeddings.weight','bert_ebd.token_type_embeddings.weight'}:
            param.requires_grad = False
            pass


    optim_params=[{'params':[p for n, p in coder_named_params
                    if not any(nd in n for nd in no_decay)],'lr':meta_lr,'weight_decay': weight_decay},
                  {'params': [p for n, p in coder_named_params
                    if any(nd in n for nd in no_decay)],'lr':meta_lr, 'weight_decay': 0.0},
                ]


    meta_optimizer=AdamW(optim_params)
    scheduler = get_linear_schedule_with_warmup(meta_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)

    # if fp16:
    #     from apex import amp
    #     model, meta_optimizer = amp.initialize(model, meta_optimizer, opt_level='O1')

    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0

    model.train()

    for it in range(train_iter):
        meta_loss, meta_right = 0.0, 0.0

        for batch in range(B):
            class_name, support, support_label, query, query_label = data_loader['train'][0]
            # Let's print out the class_name, support, support_labels, query and query_labels
            loss, right =train_one_batch(batch,class_name,support,support_label,query,query_label,model,steps,task_lr)

            meta_loss += loss
            meta_right += right

        meta_loss /= B
        meta_right /= B

        meta_optimizer.zero_grad()
        # if fp16:
        #     with amp.scale_loss(meta_loss, meta_optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        meta_loss.backward()
        meta_optimizer.step()
        scheduler.step()

        iter_loss += meta_loss
        iter_right += meta_right
        iter_sample += 1

        sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
        sys.stdout.flush()

        if (it+1)%val_step==0:
            print("")
            iter_loss, iter_right, iter_sample = 0.0,0.0,0.0
            acc = test_model(data_loader['val'], model, val_iter, steps,task_lr)
            print("")
            model.train()
            if acc > best_acc:
                print('Best checkpoint!')
                torch.save({'state_dict': model.state_dict()}, save_ckpt)

                best_acc = acc

    print("\n####################\n")
    print('Finish training model! Best acc: '+str(best_acc))


def eval_model(model,N,K,Q,eval_iter=10000,steps=30,task_lr=1e-2, noise_rate = 0,file_name=None,load_ckpt = None):
    if torch.cuda.is_available(): model = model.cuda()

    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # print('ignore {}'.format(name))
                continue
            # print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)

    accs=0.0
    model.eval()
    data_loader = FewshotDataset(file_name,N,K,Q,noise_rate)
    tot = {}
    neg = {}
    for it in range(eval_iter):
        net = model
        class_name,support,support_label,query,query_label = data_loader[0]
        _,right = train_one_batch(0,class_name, support, support_label,query,query_label,net,steps,task_lr)
        accs += right
        for i in class_name:
            if i not in tot:
                tot[i]=1
            else:
                tot[i]+=1
        if right <1:
            for i in class_name:
                if i not in neg:
                    neg[i]=1
                else:
                    neg[i]+=1
        sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')
        sys.stdout.flush()
    print("")
    print(tot)
    print(neg)
    print("")

    return accs/eval_iter

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# %%


# %%
model_name = 'PBML'
encoder_name='DeBERTa-base'

dataset2config = {"FewRel":  {"taskname":"Relation Classification",
                               "meta_lr": 5e-5,
                               "task_lr": 1e-2,
                               "weight decay": 1e-2,
                               "batch_size": 4,
                               "train_iters": 1000,
                               "steps": 30,
                               "max_length":90,
                               "warmup_step":200
                               }}

# %%
benchmark = "FewRel"  # {"FewRel","HuffPost","Reuters","Amazon"}
taskname = dataset2config[benchmark]['taskname']
meta_lr = dataset2config[benchmark]['meta_lr']
task_lr  = dataset2config[benchmark]['task_lr']
weight_decay = dataset2config[benchmark]['weight decay']
B = dataset2config[benchmark]['batch_size']
Train_iter = dataset2config[benchmark]['train_iters']
Fast_tuning_steps = dataset2config[benchmark]['steps']
max_length = dataset2config[benchmark]['max_length']
warmup_step = dataset2config[benchmark]['warmup_step']

noise_rate = 0 #  from 0 to 10

N = 5
K = 1
Q = 1

Val_iter = 2000
Val_step = 50

save_ckpt = f'./checkpoint/{benchmark}_PBML.pth'
load_ckpt = None
best_acc = 0.0

# %%
data_dir = {'benchmark': benchmark,
            'train':f'benchmark/train_wiki.json',
            'val':f'benchmark/val_wiki.json',
            'test':f'benchmark/test.json',
            'noise_rate': noise_rate,
            'candidates': f'benchmark/candidate_ebds.json',
            'pb_dropout': 0.5}

# %%
import sys

orig_stdout = sys.stdout
f = open('out_deberta_base.txt', 'w')
sys.stdout = f


print('----------------------------------------------------')
print("{}-way-{}-shot Few-Shot {}".format(N, K,taskname))
print("Model: {}".format(model_name))
print("Encoder: {}".format(encoder_name))
print('----------------------------------------------------')


start_time=time.time()

pbml=PBML(B,N,K,max_length,data_dir)

train_model(pbml,B,N,K,Q,data_dir,
            meta_lr=meta_lr,
            task_lr=task_lr,
            weight_decay = weight_decay,
            train_iter=Train_iter,
            val_iter=Val_iter,
            val_step=Val_step,
            steps = Fast_tuning_steps,
            save_ckpt = save_ckpt, load_ckpt= load_ckpt,
            best_acc = best_acc,
            warmup_step = warmup_step
            )


sys.stdout = orig_stdout
f.close()

# %%



