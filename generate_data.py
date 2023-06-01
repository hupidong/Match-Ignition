# coding: utf-8
"""
从原始文本构建模型数据集
"""
import argparse
import os

from data_loader import create_and_save_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stopword_path", type=str, default="./data/stopwords-zh.txt",
                        help="停用词文件路径")
    parser.add_argument("--data_dir", type=str, default='./data/dataset/yuqing_news/v3/orig/',
                        help="原始数据所在目录")
    parser.add_argument("--save_dir", type=str, default='./data/dataset/yuqing_news/v3/model',
                        help="转换后的模型数据存放目录")
    parser.add_argument("--from_raw_text", type=int, choices=[0, 1], default=1,
                        help="原始数据类型： 0(CNSE数据集), 1(原始文本数据)")
    parser.add_argument("--append_keyword", type=int, choices=[0, 1], default=0)

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train_data_path = os.path.join(args.data_dir, 'train.txt')
    dev_data_path = os.path.join(args.data_dir, 'dev.txt')
    test_data_path = os.path.join(args.data_dir, 'test.txt')
    train_data_save_path = os.path.join(args.save_dir, 'train.txt')
    dev_data_save_path = os.path.join(args.save_dir, "dev.txt")
    test_data_save_path = os.path.join(args.save_dir, "test.txt")

    stop_words = set()
    for w in open(args.stopword_path, 'r', encoding='utf8'):
        stop_words.add(w.strip())

    print(f"original dataset dir is {args.data_dir}")
    print(f"generated dataset save dir is {args.save_dir}")
    print(f"Create Train Set...\n")
    create_and_save_dataset(train_data_save_path, train_data_path,
                            append_title_node=True, append_title=True, append_keyword=args.append_keyword,
                            from_raw_text=args.from_raw_text, stop_words=stop_words)
    print(f"Create Validation Set...\n")
    create_and_save_dataset(dev_data_save_path, dev_data_path,
                            append_title_node=True, append_title=True, append_keyword=args.append_keyword,
                            from_raw_text=args.from_raw_text, stop_words=stop_words)
    print(f"Create Test Set...\n")
    create_and_save_dataset(test_data_save_path, test_data_path,
                            append_title_node=True, append_title=True, append_keyword=args.append_keyword,
                            from_raw_text=args.from_raw_text, stop_words=stop_words)
