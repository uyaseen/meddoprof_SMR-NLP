from utils import create_directory, join_path
from data_utils.parser import brat_to_conll
from data_utils.utils import create_train_dev_split, copy_files


def parse_data():
    tasks = ["task1", "task2"]
    tasks_desc = {"task1": "ner", "task2": "class"}
    spacy_model = "es_core_news_sm"
    # parse train set
    for task in tasks:
        print(f"*task: {task}")
        # parse all task files to a single file in BIO format
        input_dir = f"data/meddoprof-training-set 2/{task}/"
        output_dir = f"data/{task}/"
        sentences_path = join_path(output_dir, "sentences.txt")
        create_directory(output_dir)
        brat_to_conll(input_dir, sentences_path, language=spacy_model)
        # split the parsed file into train + dev set
        data_dir_1 = f"data/{task}/data-strategy=1/"
        create_directory(data_dir_1)
        train_path = join_path(data_dir_1, "train.txt")
        dev_path = join_path(data_dir_1, "dev.txt")
        create_train_dev_split(sentences_path, train_path, dev_path)
        copy_files(input_dir, dev_path)

        # parse test set
        test_dir = f"data/meddoprof-test-GS/{tasks_desc[task]}/"
        test_sentences = f"data/{task}/test.txt"
        brat_to_conll(test_dir, test_sentences, language=spacy_model)


def main():
    parse_data()


if __name__ == "__main__":
    main()
