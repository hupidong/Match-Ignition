import jieba
import json
import numpy as np
from typing import List
from transformers.data import InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import torch

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
