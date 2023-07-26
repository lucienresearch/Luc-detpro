import os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from classname import coco_novel_label_ids, OVD_COCO_ALL_CLS, coco_base_id2contiguous_id, coco_all_id2contiguous_id, coco_base_label_ids, coco_id2name, \
    COCO_CLASSES, VOC_CLASSES, Objects365_CLASSES
import coop_mini
from trainer import test_embedding, test_embedding_neg, train_epoch, get_embedding, checkpoint, accuracy1, accuracy5
from lr_scheduler import build_lr_scheduler
import math

from torch.optim import SGD

import random

BASE_CLS_NUM = 48
ALL_CLS_NUM = 65 

CLASS_NAMES_FULL = OVD_COCO_ALL_CLS
CLASS_NAMES_FULL = [x.replace('(','').replace(')','').replace('_', ' ') for x in CLASS_NAMES_FULL]
# CLASS_NAMES_FULL += ["background"]

#在COCO 65个类别中，最大的category id 是90，而 OVD_COCO_ALL_CLS 中类别个数是 65，因此必须做 id 之间的映射，否则无法与 OVD_COCO_ALL_CLS 转换的 65 个 text features 做正确的一一对应。

CLASS_NAMES = []
for id in coco_base_label_ids:
    # CLASS_NAMES.append(CLASS_NAMES_FULL[id])
    CLASS_NAMES.append(coco_id2name[id])
# CLASS_NAMES += ["background"]

def load_data_new(data_path : str, neg_thr = None, is_train = False):
    features, labels, ious = torch.load(data_path)
    
    if neg_thr == None:
        neg = (labels < 0)
    else:
        neg = (ious < neg_thr)
        labels[neg] = -1

    base = labels.new_zeros(labels.shape).bool()
    novel = labels.new_zeros(labels.shape).bool()
    for i in coco_base_label_ids: base.logical_or_(labels == i)
    for i in coco_novel_label_ids: novel.logical_or_(labels == i)
    
    # if remap:
    #     mapping = torch.Tensor(coco_base_label_ids).long()
    #     base_labels = (labels[base].view(-1,1)==mapping).int().argmax(dim=1)   #lzk: 作用？
    #     base_data = features[base], base_labels, ious[base]
    #     neg_data = features[neg], (labels).new_ones(labels[neg].shape)*BASE_CLS_NUM, ious[neg]  #lzk: 作用？
        # else:
        # base_data = features[base], labels[base], ious[base]
        # neg_data = features[neg], labels[neg], ious[neg]
    if is_train:
        # mapping = torch.Tensor(coco_base_label_ids).long()
        # d1 = labels[base].view(-1,1)
        # d2 = (d1==mapping)
        # base_labels = d2.int().argmax(dim=1)   #lzk: 作用？导致与 ori_labels 不同

        base_labels = torch.tensor([coco_base_id2contiguous_id.get(l.item(), l.item()) for l in labels[base]])
        novel_labels = torch.tensor([coco_all_id2contiguous_id.get(l.item(), l.item()) for l in labels[novel]])

        base_data = features[base], base_labels, ious[base]
        neg_data = features[neg], (labels).new_ones(labels[neg].shape)*BASE_CLS_NUM, ious[neg]  #lzk: 作用？
        novel_data = features[novel], novel_labels, ious[novel]    #lzk: 训练时并不会使用 novel_data 
    else:
        labels = torch.tensor([coco_all_id2contiguous_id.get(l.item(), l.item()) for l in labels])

        base_data = features[base], labels[base], ious[base]
        # neg_data = features[neg], labels[neg], ious[neg]
        neg_data = features[neg], (labels).new_ones(labels[neg].shape)*ALL_CLS_NUM, ious[neg]   #lzk: 这里改的对吗？
        novel_data = features[novel], labels[novel], ious[novel]    
    
    return base_data, novel_data, neg_data

def data_iou_filter(data, lb, ub):
    features, labels, ious = data
    valid = torch.logical_and(lb <= ious, ious < ub)
    return features[valid], labels[valid], ious[valid]

def get_freq(data):
    freq = [0] * 1204
    for feat, label, iou in data:
        freq[label] += 1
    return freq

def load_ens_embedding(name_list, norm = False,weight=None):
    emb = [torch.load(name, map_location='cuda').float() for name in name_list]
    if weight is not None:
        emb = [x*w / x.norm(dim=-1, keepdim = True) for x,w in zip(emb,weight)]
    else:
        emb = [x / x.norm(dim=-1, keepdim = True) for x in emb]
    emb = sum(emb)
    emb.squeeze_(dim = 0)
    if norm:
        emb = emb / emb.norm(dim = -1, keepdim = True)
    return emb

def test_neg(embedding):
    # return
    for thr in [0.5, 0.9]:
        test_embedding_neg(embedding, neg_val_dl, thr)


# novel gt test
# train_base_gt, train_set_novel_gt, _ = load_data_new('data/train_gt.pth', True)
# train_set_novel_gt = TensorDataset(*train_set_novel_gt)
# val_dlp_gt = DataLoader(train_set_novel_gt, batch_size = 1024, shuffle=True)


def sample(data, k, cats):
    feat, label, iou = data
    featk, labelk, iouk = [], [], []
    for i in cats:
        id = (label==i)
        if id.sum() == 0:
            continue
        repeat_factor = math.ceil(k/id.sum())
        # print(repeat_factor, id.sum(), feat[id].repeat([repeat_factor, 1]).shape)
        if repeat_factor == 1:
            ids = random.sample(range(id.sum()), k)
            featk.append(feat[id][ids])
            labelk.append(label[id][ids])
            iouk.append(iou[id][ids])
        else:
            featk.append(feat[id].repeat([repeat_factor-1, 1]))
            labelk.append(label[id].repeat([repeat_factor-1]))
            iouk.append(iou[id].repeat([repeat_factor-1]))
            remain = k-(repeat_factor-1)*id.sum()
            if remain > 0:
                ids = random.sample(range(id.sum()), remain)
                featk.append(feat[id][ids])
                labelk.append(label[id][ids])
                iouk.append(iou[id][ids])

            

    featk = torch.cat(featk, dim = 0)
    labelk = torch.cat(labelk, dim = 0)
    iouk = torch.cat(iouk, dim = 0)
    return featk, labelk, iouk


if __name__ == "__main__":
    # torch.random.manual_seed(825)
    # random.seed(825)

    print(sys.argv)
    # python prompt/run.py 
    # train data/xx/train data/xx/val checkpoints/exp fg_bg_5_5_6_end soft 0.5 0.5 0.6 8 end
    _, mode, train_dir, val_dir, res_dir, prefix, mode_train, bg_thr, iou_lb, iou_ub = sys.argv[:10]
    if mode == "train":
        if len(sys.argv)==10:
            ctx_num = 8 
            cls_token_position = 'end'
            neg_split = 10
        elif len(sys.argv)==11:
            ctx_num = int(sys.argv[10])
            cls_token_position = 'end'
            neg_split = 10
        elif len(sys.argv)==12:
            ctx_num = int(sys.argv[10])
            cls_token_position = sys.argv[11]
            neg_split = 10   
        else:
            ctx_num = int(sys.argv[10])
            cls_token_position = sys.argv[11]
            neg_split = sys.argv[12]
    # else:
        # _, mode, train_dir, val_dir, res_dir, prefix, mode_train, bg_thr, iou_lb, iou_ub,ctx_num = sys.argv[:11]
    if mode == "test":
        ctx_num = 8 
        cls_token_position = 'end'
        neg_split = 10
        names = sys.argv[10:]
    bg_thr, iou_lb, iou_ub = float(bg_thr), float(iou_lb), float(iou_ub)

    # Clip
    clip_model = coop_mini.load_clip_to_cpu().float()
    for params in clip_model.parameters():
        params.requires_grad_(False)
    model = coop_mini.CustomCLIP(CLASS_NAMES, clip_model, True, bg_class=(mode_train=='learn'),ctx=ctx_num,cls_token_position=cls_token_position).to('cuda')
    print('MODEL BUILD COMPLETE')


    if mode == 'train':
        train_base, train_set_novel, train_neg = load_data_new(os.path.join(train_dir, 'train_data.pth'), iou_lb, True)  
        train_base = data_iou_filter(train_base, iou_lb, iou_ub)   
        train_neg = data_iou_filter(train_neg, 0.1, bg_thr)
        # train_neg = data_iou_filter(train_neg, 0.1, 1.1)

        # k = len(train_base[0]) // BASE_CLS_NUM
        # train_base = sample(train_base, len(train_base[0]//int(neg_split)), range(BASE_CLS_NUM))
        train_neg = sample(train_neg, len(train_neg[0])//int(neg_split), [BASE_CLS_NUM])

        train_base, train_set_novel, train_neg = TensorDataset(*train_base), TensorDataset(*train_set_novel), TensorDataset(*train_neg)
        train_data = train_base if (mode_train == 'fg_only') else ConcatDataset([train_base, train_neg])       
        print("train info:", len(train_base), len(train_set_novel), len(train_neg))


        freq = get_freq(train_base)
        freq = [x / len(train_data) * BASE_CLS_NUM for x in freq]   #lzk: 为什么需要加权?
        freq = freq[:BASE_CLS_NUM]
        freq.append(2 * len(train_neg) / len(train_data)) # background   
        # freq = get_freq(train_base)
        # print("min freq =", min(freq[:BASE_CLS_NUM]))
        # freq = [x / 5e5 * BASE_CLS_NUM for x in freq]
        # freq = freq[:BASE_CLS_NUM]
        # freq.append(2 * len(train_neg) / 5e5) # background
        # print(k, k/4e5*BASE_CLS_NUM)
        # actual_cls = len(train_base) // k
        # print(actual_cls)
        # freq = [k/4e5*actual_cls] * BASE_CLS_NUM + [2*len(train_neg)/4e5]
        # freq = [1] * 867
        # print(freq)
    
        train_dl = DataLoader(train_data, batch_size = 512, shuffle=True)

    if mode == 'test':
        val_base, val_novel, _ = load_data_new(os.path.join(val_dir, 'val_data.pth'))
        _, _, val_neg = load_data_new(os.path.join(val_dir, 'val_data.pth'), iou_lb)
        val_base =  data_iou_filter(val_base, iou_ub, 1.1)
        val_novel =  data_iou_filter(val_novel, iou_ub, 1.1)
        val_neg = data_iou_filter(val_neg, bg_thr, iou_lb)
    else:
        val_base, val_novel,val_neg = load_data_new(os.path.join(val_dir, 'val_data.pth'))
    
    val_base, val_novel, val_neg = TensorDataset(*val_base), TensorDataset(*val_novel), TensorDataset(*val_neg)
    print("val info:", len(val_base), len(val_novel), len(val_neg))
    val_dl1 = DataLoader(val_base, batch_size = 1024, shuffle=True)
    val_dl2 = DataLoader(val_novel, batch_size = 1024, shuffle=True)
    # val_dlp = DataLoader(train_set_novel, batch_size = 1024, shuffle=True)
    neg_val_dl =  DataLoader(val_neg, batch_size = 1024, shuffle=True)


    if mode == 'train':

        emb = get_embedding(model, CLASS_NAMES_FULL)
        test_embedding(emb, val_dl1)
        test_embedding(emb, val_dl2)
        # test_neg(emb)

        os.makedirs(res_dir, exist_ok=True)
        # optimizer = SGD(model.prompt_learner.parameters(), lr=2e-3)
        optimizer = SGD(model.parameters(), lr=2e-3)
        scheduler = build_lr_scheduler(optimizer, 6, 0, 0)

        for i in range(6):
            print(f"epoch{i+1}")
            train_epoch(model, optimizer, train_dl, freq, mode_train)
            emb = get_embedding(model, CLASS_NAMES_FULL)
            print('val on base')
            test_embedding(emb, val_dl1)
            print('val on novel')
            test_embedding(emb, val_dl2)
            test_neg(emb)
            if emb.shape[0] > ALL_CLS_NUM:
                test_embedding(emb[:ALL_CLS_NUM], val_dl1)
                test_embedding(emb[:ALL_CLS_NUM], val_dl2)
                test_neg(emb[:ALL_CLS_NUM])
            scheduler.step()
            checkpoint(model, os.path.join(res_dir, prefix+f"epoch{i+1}"), CLASS_NAMES_FULL)
            checkpoint(model, os.path.join(res_dir, prefix+f"epoch{i+1}_empty"), [""])

        checkpoint(model, os.path.join(res_dir, prefix+f"epoch{i+1}"), CLASS_NAMES_FULL)
        # checkpoint(model, os.path.join(res_dir, prefix+f"coco"), COCO_CLASSES)
        # checkpoint(model, os.path.join(res_dir, prefix+f"voc"), VOC_CLASSES)
        # checkpoint(model, os.path.join(res_dir, prefix+f"objects365"), Objects365_CLASSES)
    
    elif mode=='test':
        # Ensemble embedding
        if names == None:
            # names = ['lvis_text_embedding.pt']
            names = ['checkpoints/exp1/test_epoch6.pth']
            # names = ['checkpoints/optim/seed456_hinge_adam_warmup_epoch7.pth']
            # names = [f'checkpoints/gen6_ens/pos{i}epoch6.pth' for i in [5,6,7,8,9]]
            # names = [f'checkpoints/gen6_ens/pos{i}voc.pth' for i in [5,6,7,8,9]]
            # names = [f'checkpoints/gen_ens/pos{i}epoch2.pth' for i in [5,6,7,8,9]]
            # names = [f'checkpoints/pos_ens/0{i}_epoch6.pth' for i in [5,6,7,8,9]]
        # if use_weight:
        ensemble_embedding = load_ens_embedding(names, norm = True)# / 5#[:ALL_CLS_NUM]
        # ensemble_embedding = load_ens_embedding(names, norm = True)# / 5#[:ALL_CLS_NUM]
        torch.save(ensemble_embedding, os.path.join(res_dir, prefix+"_ens.pth"))

        print("loaded ensemble embedding :", ensemble_embedding.shape)
        # test_embedding(ensemble_embedding[coco_base_label_ids], train_dl)
        test_embedding(ensemble_embedding, val_dl1)
        test_embedding(ensemble_embedding, val_dl2)
        # test_embedding(ensemble_embedding, neg_val_dl)
        # test_embedding(ensemble_embedding, val_dlp)
        test_neg(ensemble_embedding)

        if ensemble_embedding.shape[0] > ALL_CLS_NUM:
            print('{} only'.format(ALL_CLS_NUM))
            ensemble_embedding = ensemble_embedding[:ALL_CLS_NUM]
            # test_embedding(ensemble_embedding[coco_base_label_ids], train_dl)
            test_embedding(ensemble_embedding, val_dl1)
            test_embedding(ensemble_embedding, val_dl2)
            # test_embedding(ensemble_embedding, val_dlp)
            test_neg(ensemble_embedding)
            # test_embedding(ensemble_embedding, val_dlp_gt)
        
    elif mode=='multi':
        names = [f'checkpoints/gen6_ens/pos{i}epoch6.pth' for i in [5,6,7,8,9]]
        iou_embedding = [load_ens_embedding([name], norm = True)[:ALL_CLS_NUM] for name in names]
        # iou_embedding = torch.cat(iou_embedding)

        def test_embedding_iou(embedding, ds):
            acc1, acc5 = 0, 0
            iter = 0
            iou_thr = [.6, .7, .8, .9, 1.1]
            for feat, label, iou in ds:
                iter += 1
                if iter % 10 == 0:
                    print(iter, '/', len(ds), end='\r')


                # res = feat.to('cuda') @ embedding.t() / 0.01
                res = [feat.to('cuda') @ emb.t() / 0.01 for emb in embedding]
                res = torch.cat(res, dim = 1)
                # print(res.shape)
                # res[:,coco_base_label_ids] = -1e10
                for id, cur in enumerate(iou):
                    for tm, key in enumerate(iou_thr):
                        if cur < key:
                            break
                    label[id] = label[id] + ALL_CLS_NUM*tm
                # print(label)
                acc1 += accuracy1(res.cpu(), label)
                acc5 += accuracy5(res.cpu(), label)

            acc1 = acc1.item() / len(ds.dataset)
            acc5 = acc5.item() / len(ds.dataset)
            print(f"test acc: top1={acc1}   top5={acc5}   total={len(ds.dataset)}")
        test_embedding_iou(iou_embedding, val_dl1)
        test_embedding_iou(iou_embedding, val_dl2)
    else:
        print('unknown mode')