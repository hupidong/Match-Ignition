"""
推理代码
"""
import argparse
import json
import os.path
import random
from collections import OrderedDict
from typing import List

import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import jieba
import re
from transformers.data import InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import torch
from transformers import BertTokenizer, BertForSequenceClassification

import nlp_utils

stop_words = set()
for w in open('./data/stopwords-zh.txt'):
    stop_words.add(w.strip())


class Doc:
    def __init__(self, title_a, title_b, content_a, content_b, document_id_a, document_id_b, label=None,
                 tokenizer=None):
        self.label = label
        self.document_id_a = document_id_a
        self.document_id_b = document_id_b
        self.title_a = " ".join(tokenizer(title_a))
        self.title_b = " ".join(tokenizer(title_b))
        self.content_a = " ".join(tokenizer(content_a))
        self.content_b = " ".join(tokenizer(content_b))
        self.ner_keywords_a = ""
        self.ner_keywords_b = ""

    def parse_sentence(self, append_title_node=False):
        self.dset1 = OrderedDict()
        self.dset2 = OrderedDict()
        for idx, sent in enumerate(nlp_utils.split_chinese_sentence(self.content_a)):
            sid = '{}-{:02d}'.format(self.document_id_a, idx + 1)
            self.dset1[sid] = sent.strip()
        for idx, sent in enumerate(nlp_utils.split_chinese_sentence(self.content_b)):
            sid = '{}-{:02d}'.format(self.document_id_b, idx + 1)
            self.dset2[sid] = sent.strip()
        if append_title_node:
            sid = '{}-{:02d}'.format(self.document_id_a, 0)
            self.dset1[sid] = self.title_a.strip()
            sid = '{}-{:02d}'.format(self.document_id_b, 0)
            self.dset2[sid] = self.title_b.strip()

    def filter_word(self, stop_words):
        self.fw_dset1 = OrderedDict()
        self.fw_dset2 = OrderedDict()
        for sid in self.dset1:
            self.fw_dset1[sid] = ' '.join(list(filter(lambda x: x not in stop_words,
                                                      self.dset1[sid].split())))
        for sid in self.dset2:
            self.fw_dset2[sid] = ' '.join(list(filter(lambda x: x not in stop_words,
                                                      self.dset2[sid].split())))

    def filter_sentence(self):
        self.fs_dset1 = OrderedDict()
        self.fs_dset2 = OrderedDict()
        for sid in self.fw_dset1:
            if len(self.fw_dset1[sid].split()) >= 5:
                self.fs_dset1[sid] = self.fw_dset1[sid]
        for sid in self.fw_dset2:
            if len(self.fw_dset2[sid].split()) >= 5:
                self.fs_dset2[sid] = self.fw_dset2[sid]

    def calc_sentence_sim(self, s1, s2):
        s1 = s1.split()
        s2 = s2.split()
        return len(set(s1) & set(s2)) / (np.log(len(s1)) + np.log(len(s2)))

    def get_docid(self, sid):
        return sid.split('-')[0]

    def build_each_graph(self):
        def build_graph(dset):
            graph = nx.Graph()
            for sid in dset:
                graph.add_node(sid)
            for sid_i in dset:
                for sid_j in dset:
                    if sid_i == sid_j:
                        continue
                    sim = self.calc_sentence_sim(dset[sid_i], dset[sid_j])
                    if sim > 0:
                        graph.add_edge(sid_i, sid_j, weight=sim)
            return graph

        self.graph1 = build_graph(self.fs_dset1)
        self.node_weight_1 = nx.pagerank(self.graph1)
        self.graph2 = build_graph(self.fs_dset2)
        self.node_weight_2 = nx.pagerank(self.graph2)

    def build_pair_graph(self):
        graph = nx.Graph()
        all_sent = list(self.fs_dset1.keys()) + list(self.fs_dset2.keys())

        def get_node(sid):
            docid = self.get_docid(sid)
            if docid == self.document_id_a:
                return self.fs_dset1[sid]
            elif docid == self.document_id_b:
                return self.fs_dset2[sid]
            else:
                raise ValueError()

        for sid in all_sent:
            docid = self.get_docid(sid)
            if docid == self.document_id_a:
                graph.add_node(sid, color='red')
            elif docid == self.document_id_b:
                graph.add_node(sid, color='blue')
            else:
                raise ValueError()
        for sid_i in all_sent:
            for sid_j in all_sent:
                if sid_i == sid_j:
                    continue
                sim = self.calc_sentence_sim(get_node(sid_i), get_node(sid_j))
                if sim > 0:
                    graph.add_edge(sid_i, sid_j, weight=sim)
        self.graph = graph
        self.node_weight = nx.pagerank(self.graph)

    def show_pair_graph(self):
        node_color = [self.graph.nodes[v]['color'] for v in self.graph]
        node_size = [self.node_weight[v] * 5000 for v in self.graph]
        nx.draw(self.graph, node_color=node_color, node_size=node_size, with_labels=True)

    def show_each_graph(self):
        node_size_1 = [self.node_weight_1[v] * 5000 for v in self.graph1]
        nx.draw(self.graph1, node_size=node_size_1, with_labels=True)
        plt.show()
        node_size_2 = [self.node_weight_2[v] * 5000 for v in self.graph2]
        nx.draw(self.graph2, node_size=node_size_2, with_labels=True)
        plt.show()

    def important_sentence(self, topk=3, exclude_title=True):
        if exclude_title:
            node_t1 = '{}-{:02d}'.format(self.document_id_a, 0)
            node_t2 = '{}-{:02d}'.format(self.document_id_a, 0)
            if node_t1 in self.node_weight:
                self.node_weight[node_t1] = 0.0
            if node_t2 in self.node_weight:
                self.node_weight[node_t2] = 0.0
        imp_s1 = []
        imp_s2 = []
        for sid in self.node_weight:
            if self.get_docid(sid) == self.document_id_a:
                imp_s1.append([sid, self.node_weight[sid]])
            elif self.get_docid(sid) == self.document_id_b:
                imp_s2.append([sid, self.node_weight[sid]])

        imp_s1 = sorted(imp_s1, key=lambda x: x[1], reverse=True)
        imp_s2 = sorted(imp_s2, key=lambda x: x[1], reverse=True)
        imp_s1_sorted = sorted(imp_s1[:topk], key=lambda x: x[0])
        imp_s2_sorted = sorted(imp_s2[:topk], key=lambda x: x[0])
        return imp_s1_sorted, imp_s2_sorted

    def distinct_sentence(self, disk=3, exclude_title=True):
        if exclude_title:
            node_t1 = '{}-{:02d}'.format(self.document_id_a, 0)
            node_t2 = '{}-{:02d}'.format(self.document_id_a, 0)
            if node_t1 in self.node_weight_1:
                self.node_weight_1[node_t1] = 0.0
            if node_t2 in self.node_weight_2:
                self.node_weight_2[node_t2] = 0.0
        dist_s1 = sorted(self.node_weight_1.items(), key=lambda x: x[1], reverse=True)
        dist_s2 = sorted(self.node_weight_2.items(), key=lambda x: x[1], reverse=True)
        dist_s1_sorted = sorted(dist_s1[:disk], key=lambda x: x[0])
        dist_s2_sorted = sorted(dist_s2[:disk], key=lambda x: x[0])
        return dist_s1_sorted, dist_s2_sorted

    def selected_sentence_1(self, disk=1, topk=3, exclude_title=True):
        dist_s1, dist_s2 = self.distinct_sentence(disk, exclude_title)
        for k, v in dist_s1:
            self.node_weight[k] += 10
        for k, v in dist_s2:
            self.node_weight[k] += 10
        results = self.important_sentence(topk, exclude_title)
        for k, v in dist_s1:
            self.node_weight[k] -= 10
        for k, v in dist_s2:
            self.node_weight[k] -= 10
        return results

    def selected_sentence_2(self, disk=3, topk=1, exclude_title=True):
        imp_s1, imp_s2 = self.important_sentence(topk, exclude_title)
        for k, v in imp_s1:
            self.node_weight_1[k] += 10
        for k, v in imp_s2:
            self.node_weight_2[k] += 10
        results = self.distinct_sentence(disk, exclude_title)
        for k, v in imp_s1:
            self.node_weight_1[k] -= 10
        for k, v in imp_s2:
            self.node_weight_2[k] -= 10
        return results


def create_dataset(doc_pair: Doc, append_title_node=False, append_title=False, append_keyword=False, stop_words=None):
    doc_pair.parse_sentence(append_title_node=append_title_node)
    doc_pair.filter_word(stop_words)
    doc_pair.filter_sentence()
    doc_pair.build_pair_graph()
    # doc.build_each_graph()
    # s1 = s2 = []
    s1, s2 = doc_pair.important_sentence(5)
    # s1, s2 = doc.selected_sentence_1(disk=3, topk=5)
    # s1, s2 = doc.selected_sentence_2(disk=5, topk=3)
    # s1, s2 = list(doc.dset1.keys())[:7], list(doc.dset2.keys())[:7]
    # s1 = [[x, 1] for x in s1]
    # s2 = [[x, 1] for x in s2]
    d1 = []
    d2 = []
    if append_title:
        d1.append(doc_pair.title_a + ' ☢')
        d2.append(doc_pair.title_b + ' ☢')
    if append_keyword:
        d1.append(doc_pair.ner_keywords_a + ' ☄')
        d2.append(doc_pair.ner_keywords_b + ' ☄')
    for s in s1:
        d1.append(doc_pair.dset1[s[0]])
    for s in s2:
        d2.append(doc_pair.dset2[s[0]])
    # for s in s1:
    #    d1.append(' '.join(['的'] * len(''.join(doc.dset1[s[0]].split()))))
    # for s in s2:
    #    d2.append(' '.join(['的'] * len(''.join(doc.dset2[s[0]].split()))))
    if (len(d1) == 0 or len(d2) == 0) and doc_pair.label == 1:
        raise ('Error')

    d1 = ' '.join(d1)
    d2 = ' '.join(d2)
    if len(d1) == 0:
        d1 = '龎'
    if len(d2) == 0:
        d2 = '龎'
    return (d1, d2, doc_pair.label)


def predict(model,
            tokenizer,
            titles_a: [List, str],
            titles_b: [List, str],
            contents_a: [List, str],
            contents_b: [List, str],
            uids_a: [List, str],
            uids_b: [List, str],
            **kwargs):
    assert type(titles_a) == type(contents_a) == type(titles_b) == type(contents_b) == type(uids_a) == type(uids_b)
    if isinstance(titles_a, list):
        assert len(titles_a) == len(contents_a) == len(titles_b) == len(
            contents_b) == len(uids_a) == len(uids_b), "length of titles and contents should be equal"

    max_tokens = kwargs.get("max_tokens", 512)
    device = kwargs.get("device", "cpu")
    if isinstance(titles_a, str):
        titles_a = [titles_a]
        titles_b = [titles_b]
        contents_a = [contents_a]
        contents_b = [contents_b]
        uids_a = [uids_a]
        uids_b = [uids_b]

    model_inputs = create_inputs(titles_a=titles_a,
                                 titles_b=titles_b,
                                 contents_a=contents_a,
                                 contents_b=contents_b,
                                 docs_id_a=uids_a,
                                 docs_id_b=uids_b,
                                 tokenizer=tokenizer,
                                 max_tokens=max_tokens,
                                 device=device)
    outputs = model(**model_inputs)
    _, logits = outputs[:2]
    probs = torch.softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()
    labels = np.argmax(probs, axis=1)
    return probs, labels


def create_inputs(titles_a, titles_b, contents_a, contents_b, docs_id_a, docs_id_b, tokenizer, max_tokens=400,
                  device="cpu"):
    kw_token_id, title_token_id, empty_token_id = tokenizer.convert_tokens_to_ids(['☄', '☢', '龎'])
    examples = []

    for title_a, title_b, content_a, content_b, doc_id_a, doc_id_b in zip(titles_a, titles_b, contents_a, contents_b,
                                                                          docs_id_a, docs_id_b):
        doc_pair = Doc(title_a=title_a,
                       title_b=title_b,
                       content_a=content_a,
                       content_b=content_b,
                       document_id_a=doc_id_a,
                       document_id_b=doc_id_b,
                       tokenizer=jieba.lcut,
                       label="1"
                       )
        text_a, text_b, label = create_dataset(doc_pair,
                                               append_title_node=True,
                                               append_title=True,
                                               append_keyword=False,
                                               stop_words=stop_words
                                               )
        guid = doc_id_a + "-" + doc_id_b
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=["0", "1"],
        max_length=max_tokens,
        output_mode='classification',
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=device)
    all_gate_mask = []
    for idx, f in enumerate(features):
        input_ids_t = np.array(f.input_ids)
        # print(f.input_ids)
        s1_sep, s2_sep = np.where(input_ids_t == tokenizer.sep_token_id)[0]
        s1_title, s2_title = np.where(input_ids_t == title_token_id)[0]
        gate_mask = np.zeros(len(f.input_ids), dtype=np.float32)
        gate_mask[:s1_title + 1] = 1
        gate_mask[s1_sep:s2_title + 1] = 1
        gate_mask[s2_sep] = 1
        all_gate_mask.append(gate_mask)
    all_gate_mask = torch.tensor(all_gate_mask, device=device)[:, :, None]
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long, device=device)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long, device=device)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long, device=device)

    return {"input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "token_type_ids": all_token_type_ids,
            "gate_mask": all_gate_mask,
            "labels": all_labels
            }


def read_examples(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        return lines


def replace_invisible_space(text, repl=" "):
    """
    替换非可见字符
    :param text: 原始文本
    :param repl: 非可见字符替换成的字符
    :return:替换后的文本
    """
    text = re.sub(
        r'[\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u200C\u200D\u202F\u205F\u00A0\u3000\ufeff]',
        repl, text)
    return text


def clean_text(text, remove_space=True):
    # clean url
    pattern_url = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    text = re.sub(pattern=pattern_url, repl="", string=text)
    # clean space
    text = replace_invisible_space(text, " ")
    if remove_space:
        text = text.replace(" ", "")

    return text


if __name__ == "__main__":
    from sklearn.metrics import precision_recall_fscore_support, classification_report

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/bert_imp_sign_pr90_tr/checkpoint-14000")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str, default="test_data/news_matching_pairs_ratio_of_neg2pos=10.jsonl")
    parser.add_argument("--batch_size", type=int, default=100)

    args = parser.parse_args()
    args.save_path_jsonl = os.path.splitext(args.data_path)[0] + "_eval.jsonl"
    args.save_path_excel = os.path.splitext(args.data_path)[0] + "_eval.xlsx"
    print(f"save_path_jsonl: {args.save_path_jsonl}")
    print(f"save_path_excel: {args.save_path_excel}")

    len_reduce_list = [int(400 * (0.90) ** i) for i in range(1, 13)]

    gate_type = 'pagerank'

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(
        args.model_path,
        output_attentions=True,
        len_reduce_list=len_reduce_list,
        gate_type=gate_type)
    model.to(args.device)
    model.eval()

    uids_a = "dbec31a4c499dce1"
    uids_b = "efb2c2eec3b0fe93"
    titles_a = "广东银监局行政处罚信息公开表(2018年13号)"
    contents_a = "罚款80万元具体原因如下：流动资金贷款业务严重违反审慎经营规则"
    titles_b = "美国宣布新制裁5所中国高校，其中2所助力“嫦五”登月！清华北大仍旧沉默"
    contents_b = "美帝国主义在国际上向来是飞扬跋扈、目中无人，只要不符合美国普世价值观的人和事都要受到所谓的“制裁”。在2020年12月18日之前，中国大陆共计有13所高校被美国列入实体清单，也就是常说的“制裁”，其中包括：985高校10所：北京航空航天大学、哈尔滨工业大学，西北工业大学、国防科技大学、湖南大学、同济大学、西安交通大学、电子科技大学、中国人民大学、四川大学211院校2所：南昌大学、哈尔滨工程大学。非211院校1所：广东工业大学　　无独有偶，近日，12月18日美国商务部宣布正式“制裁”中芯国际，同时会在12月22日官方公布新的对华77个实体及个人的“制裁”名单。通过整理发现，又有5所中国高校光荣上榜，它们分别是北京理工大学、北京邮电大学、南京航空航天大学，南京理工大学和天津大学。　　其中，南京航空航天大学和天津大学，在此次嫦娥五号探月任务中扮演了重要角色。南京航空航天大学航天学院院长叶培建院士担任嫦娥系列探测器总指挥、总设计师顾问；天津大学空间力学团队在嫦娥五号着陆器稳定着陆、返回大气层的稳定性和姿态分析上做出了重大贡献。　　此外，北京理工大学、北京邮电大学和南京理工大学要么是具有一定的红色军工背景，要么为中国的5G移动通信做出了贡献。这样看来，美帝国主义越是“制裁”这些高校，越证明了她们为新时代中国特色社会主义建设贡献了巨大的力量。　　然而，处于我国高等教育殿堂巅峰的清华大学和北京大学依旧沉默，仍然没有出现在这份“制裁”名单上，同时也没有师生在嫦娥五号事业中大放光彩，这不禁让人纳闷，清华北大的名校生都去哪里了呢？难道真如施一公所说，清北一直在为美国输送人才？　　目前，拜登已经赢得总统选举，特朗普政府下台只是时间问题。此次“制裁”是否意味着特朗普已经昏招频频、狗急跳墙？要知道，没有任何人和事能够阻挡中华民族伟大复兴的前进步伐！特别声明：以上内容(如有图片或视频亦包括在内)为自媒体平台“网易号”用户上传并发布，本平台仅提供信息存储服务。Notice: The content above (including the pictures and videos if any) is uploaded and posted by a user of NetEase Hao, which is a social media platform and only provides information storage services."

    examples = read_examples(args.data_path)

    random.seed(1000)
    random.shuffle(examples)

    eval_examples = examples[0:]
    goldens = [example["label"] for example in eval_examples]
    preds = []
    # make batches
    batch_num = math.ceil(len(eval_examples) / args.batch_size)
    batches = [eval_examples[i * args.batch_size: (i + 1) * args.batch_size] for i in range(batch_num)]
    for batch_idx, batch in enumerate(tqdm(batches)):
        titles_a = [clean_text(example["doc_a"]["title"]) for example in batch]
        contents_a = [clean_text(example["doc_a"]["content"]) for example in batch]
        uids_a = [example["doc_a"]["uid"] for example in batch]

        titles_b = [clean_text(example["doc_b"]["title"]) for example in batch]
        contents_b = [clean_text(example["doc_b"]["content"]) for example in batch]
        uids_b = [example["doc_b"]["uid"] for example in batch]
        for example, title_a, content_a, title_b, content_b in zip(batch, titles_a, contents_a, titles_b, contents_b):
            example["doc_a"]["title"] = title_a
            example["doc_a"]["content"] = content_a
            example["doc_b"]["title"] = title_b
            example["doc_b"]["content"] = content_b

        probs, labels = predict(model=model,
                                tokenizer=tokenizer,
                                titles_a=titles_a,
                                titles_b=titles_b,
                                contents_a=contents_a,
                                contents_b=contents_b,
                                uids_a=uids_a,
                                uids_b=uids_b,
                                max_tokens=args.max_tokens,
                                device=args.device)
        for example, prob, label in zip(batch, probs, labels):
            label = int(label)
            prob = float(prob[label])
            example["pred_info"] = {"label": label,
                                    "prob": prob}
            example["pred"] = label
            example["is_correct"] = label == example["label"]
            preds.append(label)

    print(classification_report(goldens, preds))
    with open(args.save_path_jsonl, 'w', encoding='utf8') as f:
        for example in eval_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    df_examples = pd.DataFrame(data=eval_examples)
    df_examples.to_excel(args.save_path_excel)
