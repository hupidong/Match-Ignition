"""
推理代码
"""
import argparse
import json
import os.path
import random
from typing import List
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import jieba
import re
from transformers.data import InputExample

from nlp_utils import clean_text
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from data_loader import Doc, create_single_example

stop_words = set()
for w in open('./data/stopwords-zh.txt'):
    stop_words.add(w.strip())


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
                                 tokenizer_bert=tokenizer,
                                 max_tokens=max_tokens,
                                 device=device)
    outputs = model(**model_inputs)
    _, logits = outputs[:2]
    probs = torch.softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()
    labels = np.argmax(probs, axis=1)
    return probs, labels


def create_inputs(titles_a,
                  titles_b,
                  contents_a,
                  contents_b,
                  docs_id_a,
                  docs_id_b,
                  tokenizer_bert,
                  tokenizer_word=jieba.lcut,
                  max_tokens=512,
                  device="cpu"):
    kw_token_id, title_token_id, empty_token_id = tokenizer_bert.convert_tokens_to_ids(['☄', '☢', '龎'])
    examples = []

    for title_a, title_b, content_a, content_b, doc_id_a, doc_id_b in zip(titles_a, titles_b, contents_a, contents_b,
                                                                          docs_id_a, docs_id_b):
        doc_pair = Doc(title_a=title_a,
                       title_b=title_b,
                       content_a=content_a,
                       content_b=content_b,
                       doc_id_a=doc_id_a,
                       doc_id_b=doc_id_b,
                       tokenizer=tokenizer_word,
                       label="1"
                       )
        text_a, text_b, label = create_single_example(doc_pair,
                                                      append_title_node=True,
                                                      append_title=True,
                                                      append_keyword=False,
                                                      stop_words=stop_words
                                                      )
        guid = doc_id_a + "-" + doc_id_b
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    features = convert_examples_to_features(
        examples,
        tokenizer_bert,
        label_list=["0", "1"],
        max_length=max_tokens,
        output_mode='classification',
        pad_token=tokenizer_bert.convert_tokens_to_ids([tokenizer_bert.pad_token])[0],
        pad_token_segment_id=0,
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=device)
    all_gate_mask = []
    for idx, f in enumerate(features):
        input_ids_t = np.array(f.input_ids)
        # print(f.input_ids)
        s1_sep, s2_sep = np.where(input_ids_t == tokenizer_bert.sep_token_id)[0]
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


if __name__ == "__main__":
    from sklearn.metrics import precision_recall_fscore_support, classification_report

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/yuqing_news_v0/bert-base-chinese/checkpoint-527")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--data_path", type=str, default="data/dataset/yuqing_news/v0/orig/test.txt")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    args.save_path_jsonl = os.path.splitext(args.data_path)[0] + "_eval.jsonl"
    args.save_path_excel = os.path.splitext(args.data_path)[0] + "_eval.xlsx"

    print(f"model_path: {args.model_path}")
    print(f"eval_data_path: {args.data_path}")
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
