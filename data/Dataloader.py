import json
import torch
import copy
from data.Dialog import *
import numpy as np
import re


def read_corpus(file):
    with open(file, mode='r', encoding='UTF8') as infile:
        info = infile.readline()
        data = json.loads(info)

        instances = []
        for dialog_info in data:
            instance = info2instance(dialog_info)
            instances.append(instance)
        return instances

def info2instance(dialog_info):
    instance = Dialog()

    instance.id = dialog_info["id"]
    instance.original_EDUs = copy.deepcopy(dialog_info["edus"])

    root_edu = dict()
    root_edu['text'] = "<root>"
    root_edu['speaker'] = "<root>"
    root_edu['tokens'] = ["<root>"]

    instance.EDUs.append(root_edu)
    instance.EDUs += dialog_info["edus"]
    instance.id = dialog_info["id"]
    instance.relations = dialog_info["relations"]

    instance.real_relations = [[] for idx in range(len(instance.original_EDUs))]

    rel_matrix = np.zeros([len(instance.original_EDUs), len(instance.original_EDUs)]) ## arc flag
    for relation in instance.relations:
        index = relation['y']
        head = relation['x']
        if rel_matrix[index, head] >= 1: continue
        if head > index: continue
        if index >= len(instance.real_relations): continue
        if head >= len(instance.real_relations): continue

        rel_matrix[index, head] += 1
        instance.real_relations[index].append(relation)

    instance.sorted_real_relations = []
    for idx, rel_relation in enumerate(instance.real_relations):
        r = sorted(rel_relation,  key=lambda rel_relation:rel_relation['x'], reverse=False)
        instance.sorted_real_relations.append(r)

    instance.gold_arcs = [[] for idx in range(len(instance.EDUs))]
    instance.gold_rels = [[] for idx in range(len(instance.EDUs))]

    instance.gold_arcs[0].append(-1)
    instance.gold_rels[0].append('<root>')
    for idx, relation_list in enumerate(instance.sorted_real_relations):
        if len(relation_list) > 0:
            relation = relation_list[0]
            rel = relation['type']
            index = relation['y'] + 1
            head = relation['x'] + 1
            if head > index: continue
            instance.gold_arcs[index].append(head)
            instance.gold_rels[index].append(rel)
    for idx, arc in enumerate(instance.gold_arcs):
        if len(arc) == 0:
            instance.gold_arcs[idx].append(0)
            instance.gold_rels[idx].append('<root>')

    for arc in instance.gold_arcs:
        assert len(arc) == 1
    for rel in instance.gold_rels:
        assert len(rel) == 1

    for EDU in instance.EDUs:
        tokens = EDU['tokens']
        for idx, token in enumerate(tokens):
            if re.match("\d+", token):
                token = '[num]'
            tokens[idx] = token.lower()

    for idx, cur_EDU in enumerate(instance.EDUs):
        if idx == 0:
            turn = 0
        else:
            last_EDU = instance.EDUs[idx - 1]
            if last_EDU["speaker"] != cur_EDU["speaker"]:
                turn += 1
        cur_EDU["turn"] = turn

    return instance


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def batch_label_variable(onebatch, vocab):
    batch_gold_arcs = []
    batch_gold_rels = []
    for idx, instance in enumerate(onebatch):
        gold_arcs = np.zeros([len(instance.gold_arcs)])
        gold_rels = np.zeros([len(instance.gold_arcs)])
        for idy, gold_arc in enumerate(instance.gold_arcs):
            gold_arcs[idy] = instance.gold_arcs[idy][0]

        for idy, gold_rel in enumerate(instance.gold_rels):
            rel = instance.gold_rels[idy][0]
            if idy == 0:
                gold_rels[idy] = -1
            else:
                gold_rels[idy] = vocab.rel2id(rel)
        batch_gold_arcs.append(gold_arcs)
        batch_gold_rels.append(gold_rels)
    return batch_gold_arcs, batch_gold_rels



def batch_data_variable(onebatch, vocab):
    batch_size = len(onebatch)
    edu_lengths = [len(instance.EDUs) for instance in onebatch]
    max_edu_len = max(edu_lengths)
    arc_masks = np.zeros([batch_size, max_edu_len, max_edu_len])

    max_word_len = max(len(EDU['tokens']) for instance in onebatch for EDU in instance.EDUs)

    word_lengths = np.ones([batch_size, max_edu_len], dtype=int)
    word_indexs = np.zeros([batch_size, max_edu_len, max_word_len], dtype=int)
    extword_indexs = np.zeros([batch_size, max_edu_len, max_word_len], dtype=int)

    for idx, instance in enumerate(onebatch):
        for idy, EDU in enumerate(instance.EDUs):
            words = EDU['tokens']
            word_lengths[idx, idy] = len(words)
            for idz, word in enumerate(words):
                word_indexs[idx, idy, idz] = vocab.word2id(word)
                extword_indexs[idx, idy, idz] = vocab.extword2id(word)

        edu_len = len(instance.EDUs)
        for idy in range(edu_len):
            for idz in range(idy):
                arc_masks[idx, idy, idz] = 1.

    word_indexs = torch.tensor(word_indexs)
    extword_indexs = torch.tensor(extword_indexs)
    arc_masks = torch.tensor(arc_masks)
    word_lengths = word_lengths.flatten()

    return word_indexs, extword_indexs, word_lengths, edu_lengths, arc_masks

def batch_feat_variable(onebatch, vocab):
    batch_size = len(onebatch)
    edu_lengths = [len(instance.EDUs) for instance in onebatch]
    max_edu_len = max(edu_lengths)
    diaglog_feats = np.ones([batch_size, max_edu_len, max_edu_len, 3])

    for idx, instance in enumerate(onebatch):
        edu_len = len(instance.EDUs)
        for idy in range(edu_len):
            for idz in range(idy):
                diaglog_feats[idx, idy, idz, 0] = (idy - idz)
                diaglog_feats[idx, idy, idz, 1] = (instance.EDUs[idy]["speaker"] == instance.EDUs[idz]["speaker"])
                diaglog_feats[idx, idy, idz, 2] = (instance.EDUs[idy]["turn"] == instance.EDUs[idz]["turn"])

    diaglog_feats = torch.tensor(diaglog_feats).type(torch.FloatTensor)
    return diaglog_feats