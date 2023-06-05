# Match-Ignition
> PyTorch implementation of [CIKM 2021] Match-Ignition: Plugging PageRank into Transformer for Long-form Text Matching.

[![Python 3.6](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## Usage

### Environment Preparation
```bash
pip install -r requirements.txt
cd transformers-v4.30.0.dev0
pip install -e .
```

### Data Preparation
```bash
cd data/dataset/cnse
tar xzvf orig.tar.gz
```
Note: the original dataset can be downloaded from [here](https://github.com/BangLiu/ArticlePairMatching).

### Sentence-level Noise Filtering
```bash
# CNSE dataset
python generate_data.py --data_dir=data/dataset/cnse/orig --save_dir==data/dataset/cnse/model --from_raw_text=0 --append_keyword=1

# general raw-text dataset
python generate_data.py --data_dir=data/dataset/yuqing_news/example/orig --save_dir=data/dataset/yuqing_news/example/model --from_raw_text=1 --append_keyword=0
```

### Word-level Noise Filtering
```bash
python run.py 
```

## Citation

If you use Match-Ignition in your research, please use the following BibTex entry.

```
@inproceedings{pang2021matchignition,
    title={Match-Ignition: Plugging PageRank into Transformer for Long-form Text Matching},
    author={Liang Pang and Yanyan Lan and Xueqi Cheng},
    booktitle = {Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
    series = {CIKM'21},
    year = {2021},
}
```

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

Copyright (c) 2019-present, Liang Pang (pl8787)
