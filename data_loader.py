import json
import os

import jieba
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from transformers.data import DataProcessor, InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from typing import List
from joblib import Parallel, delayed
import nlp_utils

DOC_SEP = "\t\t\t\t"

n_jobs = 10


class Doc:
    """
    兼容原半结构化数据(CNSE(chinese news same event dataset))初始化和原始非结构化文本初始化（from-scratch）
    """

    def __init__(self,
                 text=None,
                 title_a="",
                 title_b="",
                 content_a="",
                 content_b="",
                 doc_id_a="",
                 doc_id_b="",
                 label=None,
                 tokenizer=None):
        if text is not None:
            part = text.split('|')
            self.label = int(part[0])
            self.doc_id_a = part[1]
            self.doc_id_b = part[2]
            self.title_a = part[3]
            self.title_b = part[4]
            self.content_a = part[5]
            self.content_b = part[6]
            self.ner_keywords_a = ' '.join(part[11].split(',')[:8])
            self.ner_keywords_b = ' '.join(part[12].split(',')[:8])
        else:
            self.label = label
            self.doc_id_a = doc_id_a
            self.doc_id_b = doc_id_b
            self.title_a = " ".join(tokenizer(title_a))
            self.title_b = " ".join(tokenizer(title_b))
            self.content_a = " ".join(tokenizer(content_a))
            self.content_b = " ".join(tokenizer(content_b))
            # TODO extrct ner_keywords from document
            self.ner_keywords_a = ""
            self.ner_keywords_b = ""

    def parse_sentence(self, append_title_node=False):
        self.dset1 = OrderedDict()
        self.dset2 = OrderedDict()
        for idx, sent in enumerate(nlp_utils.split_chinese_sentence(self.content_a)):
            sid = '{}-{:02d}'.format(self.doc_id_a, idx + 1)
            self.dset1[sid] = sent.strip()
        for idx, sent in enumerate(nlp_utils.split_chinese_sentence(self.content_b)):
            sid = '{}-{:02d}'.format(self.doc_id_b, idx + 1)
            self.dset2[sid] = sent.strip()
        if append_title_node:
            sid = '{}-{:02d}'.format(self.doc_id_a, 0)
            self.dset1[sid] = self.title_a.strip()
            sid = '{}-{:02d}'.format(self.doc_id_b, 0)
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
            if docid == self.doc_id_a:
                return self.fs_dset1[sid]
            elif docid == self.doc_id_b:
                return self.fs_dset2[sid]
            else:
                raise ValueError()

        for sid in all_sent:
            docid = self.get_docid(sid)
            if docid == self.doc_id_a:
                graph.add_node(sid, color='red')
            elif docid == self.doc_id_b:
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
            node_t1 = '{}-{:02d}'.format(self.doc_id_a, 0)
            node_t2 = '{}-{:02d}'.format(self.doc_id_a, 0)
            if node_t1 in self.node_weight:
                self.node_weight[node_t1] = 0.0
            if node_t2 in self.node_weight:
                self.node_weight[node_t2] = 0.0
        imp_s1 = []
        imp_s2 = []
        for sid in self.node_weight:
            if self.get_docid(sid) == self.doc_id_a:
                imp_s1.append([sid, self.node_weight[sid]])
            elif self.get_docid(sid) == self.doc_id_b:
                imp_s2.append([sid, self.node_weight[sid]])

        imp_s1 = sorted(imp_s1, key=lambda x: x[1], reverse=True)
        imp_s2 = sorted(imp_s2, key=lambda x: x[1], reverse=True)
        imp_s1_sorted = sorted(imp_s1[:topk], key=lambda x: x[0])
        imp_s2_sorted = sorted(imp_s2[:topk], key=lambda x: x[0])
        return imp_s1_sorted, imp_s2_sorted

    def distinct_sentence(self, disk=3, exclude_title=True):
        if exclude_title:
            node_t1 = '{}-{:02d}'.format(self.doc_id_a, 0)
            node_t2 = '{}-{:02d}'.format(self.doc_id_a, 0)
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


def create_single_example_from_doc_pair(doc_pair: Doc,
                                        append_title_node=False,
                                        append_title=False,
                                        append_keyword=False,
                                        stop_words=set()):
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


def create_and_save_dataset(filepath,
                            datapath,
                            append_title_node=False,
                            append_title=False,
                            append_keyword=False,
                            stop_words=set(),
                            from_raw_text=False,
                            tokenizer_word=jieba.lcut):
    fout = open(filepath, 'w', encoding='utf8')
    for line in tqdm(open(datapath, 'r', encoding='utf8')):
        d1, d2, doc = create_single_sample(line=line,
                                           append_title_node=append_title_node,
                                           append_title=append_title,
                                           append_keyword=append_keyword,
                                           stop_words=stop_words,
                                           from_raw_text=from_raw_text,
                                           tokenizer_word=tokenizer_word
                                           )
        if d1 is None and d2 is None and doc is None:
            continue
        fout.write(f'{d1}{DOC_SEP}{d2}{DOC_SEP}{doc.label}\n')
    fout.close()


def create_single_sample(line, append_title_node, append_title, append_keyword,
                         stop_words, from_raw_text, tokenizer_word):
    if from_raw_text:
        example = json.loads(line)
        title_a = nlp_utils.clean_text(example["doc_a"]["title"])
        content_a = nlp_utils.clean_text(example["doc_a"]["content"])
        uid_a = example["doc_a"]["uid"]

        title_b = nlp_utils.clean_text(example["doc_b"]["title"])
        content_b = nlp_utils.clean_text(example["doc_b"]["content"])
        uid_b = example["doc_b"]["uid"]
        doc = Doc(title_a=title_a,
                  title_b=title_b,
                  content_a=content_a,
                  content_b=content_b,
                  doc_id_a=uid_a,
                  doc_id_b=uid_b,
                  label=example["label"],
                  tokenizer=tokenizer_word)
    else:
        doc = Doc(line)
    doc.parse_sentence(append_title_node=append_title_node)
    doc.filter_word(stop_words)
    doc.filter_sentence()
    doc.build_pair_graph()
    # doc.build_each_graph()
    # s1 = s2 = []
    s1, s2 = doc.important_sentence(5)
    # s1, s2 = doc.selected_sentence_1(disk=3, topk=5)
    # s1, s2 = doc.selected_sentence_2(disk=5, topk=3)
    # s1, s2 = list(doc.dset1.keys())[:7], list(doc.dset2.keys())[:7]
    # s1 = [[x, 1] for x in s1]
    # s2 = [[x, 1] for x in s2]
    d1 = []
    d2 = []
    if append_title:
        d1.append(doc.title_a + ' ☢')
        d2.append(doc.title_b + ' ☢')
    if append_keyword:
        d1.append(doc.ner_keywords_a + ' ☄')
        d2.append(doc.ner_keywords_b + ' ☄')
    for s in s1:
        d1.append(doc.dset1[s[0]])
    for s in s2:
        d2.append(doc.dset2[s[0]])
    # for s in s1:
    #    d1.append(' '.join(['的'] * len(''.join(doc.dset1[s[0]].split()))))
    # for s in s2:
    #    d2.append(' '.join(['的'] * len(''.join(doc.dset2[s[0]].split()))))
    if (len(d1) == 0 or len(d2) == 0) and doc.label == 1:
        print('Error')
        return None, None, None
    d1 = ' '.join(d1)
    d2 = ' '.join(d2)
    if len(d1) == 0:
        d1 = '龎'
    if len(d2) == 0:
        d2 = '龎'
    return (d1, d2, doc)


class MatchingDataProcessor(DataProcessor):
    """Processor for the Matching data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(os.path.join(data_dir, "train.txt")))
        return self._create_examples(os.path.join(data_dir, "train.txt"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(os.path.join(data_dir, "dev.txt")))
        return self._create_examples(os.path.join(data_dir, "dev.txt"), "dev")

    def get_test_examples(self, data_dir):
        print("LOOKING AT {}".format(os.path.join(data_dir, "test.txt")))
        return self._create_examples(os.path.join(data_dir, "test.txt"), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, filename, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(open(filename)):
            line = line.strip()
            line = line.replace('[EOS]', '').replace('[SEP]', '').replace('[KW]', '')
            part = line.split(DOC_SEP)
            part = [''.join(x.split()) for x in part]
            guid = "%s-%s" % (set_type, i)
            text_a = part[0]
            text_b = part[1]
            label = part[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# In[ ]:
def process_for_long_doc(example, max_len, tokenizer, title_token_id, max_title=32):
    """
    对于长文本，要特殊处理一下，保证两篇文章都能取到足够的字符长度
    """
    eps = 1
    # 最大字符级长度相比token可以适当膨胀一下，一般经验参数1.3左右，根据不同vocab可能会有不同
    dilatation_coref = 1.2
    title_token = tokenizer.convert_ids_to_tokens(title_token_id)
    text_a = example.text_a
    text_b = example.text_b
    title_a, content_a = text_a.split(title_token)
    title_b, content_b = text_b.split(title_token)
    title_a = title_a[0: max_title]
    title_b = title_b[0: max_title]
    max_len_char = dilatation_coref * max_len
    len_content_a = int(len(content_a) / (len(content_a) + len(content_b) + eps) * (
            max_len_char - len(title_a) - len(title_b)))
    len_content_b = int(len(content_b) / (len(content_a) + len(content_b) + eps) * (
            max_len_char - len(title_a) - len(title_b)))
    content_a = content_a[0: len_content_a]
    content_b = content_b[0: len_content_b]
    example.text_a = title_a + title_token + content_a
    example.text_b = title_b + title_token + content_b
    return example


def load_and_cache_examples(data_dir, max_len, mode, tokenizer, title_token_id, use_cache=False):
    processor = MatchingDataProcessor()

    # load data
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}".format(
            mode,
            str(max_len),
        ),
    )
    if os.path.exists(cached_features_file) and use_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        if mode == 'train':
            examples = (
                processor.get_train_examples(data_dir)
            )
        elif mode == 'dev':
            examples = (
                processor.get_dev_examples(data_dir)
            )
        else:
            examples = (
                processor.get_test_examples(data_dir)
            )
        # long-document
        examples = [process_for_long_doc(example, max_len, tokenizer, title_token_id) for example in examples]
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=processor.get_labels(),
            max_length=max_len,
            output_mode='classification'
            # pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            # pad_token_segment_id=0,
        )
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_gate_mask = []
    for idx, f in enumerate(features):
        input_ids_t = np.array(f.input_ids)
        # print(f.input_ids)
        s1_sep, s2_sep = np.where(input_ids_t == tokenizer.sep_token_id)[0]
        # TODO how to use ner-keywords??
        # s1_kw, s2_kw = np.where(input_ids_t == kw_token_id)[0]
        try:
            s1_title, s2_title = np.where(input_ids_t == title_token_id)[0]
        except Exception as e:
            print(f"{idx}")
            print(f"{e.__str__()}")
            raise e
        gate_mask = np.zeros(len(f.input_ids), dtype=np.float32)
        gate_mask[:s1_title + 1] = 1
        gate_mask[s1_sep:s2_title + 1] = 1
        gate_mask[s2_sep] = 1
        # print(s1_sep, s2_sep)
        # print(s1_kw, s2_kw)
        # print(s1_kw, s2_kw-s1_sep)
        # for i in range(len(gate_mask)):
        #    print('{}:{}'.format(input_ids_t[i], gate_mask[i]), end=' ')
        # print('')
        # input()
        # print('\r{}'.format(idx), end='')
        all_gate_mask.append(gate_mask)
    all_gate_mask = torch.tensor(all_gate_mask)[:, :, None]
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_gate_mask)

    return dataset
