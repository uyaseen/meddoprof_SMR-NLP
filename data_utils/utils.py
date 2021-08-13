import codecs
from shutil import copy2
from utils import create_directory, join_path, dirname


def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, "r", "utf8"):
        line = line.strip()
        if not line:
            if len(sentence) > 0:
                if "DOCSTART" not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            line_sp = line.split()
            assert len(line_sp) >= 2
            sentence.append(line_sp)
    if len(sentence) > 0:
        if "DOCSTART" not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def write_sentences(sentences, path):
    with codecs.open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for line in sentence:
                f.write("{}\n".format(" ".join(line)))
            f.write("\n")

    print("{} created".format(path))


def create_train_dev_split(sentences_path, train_path, dev_path, dev_percent=0.1):
    all_sentences = load_sentences(sentences_path)
    dev_idx = int(len(all_sentences) * dev_percent)
    dev_sentences = all_sentences[0:dev_idx]
    train_sentences = all_sentences[dev_idx:]
    write_sentences(train_sentences, train_path)
    write_sentences(dev_sentences, dev_path)


def get_doc_annotation_files(dev_path, doc_idx=2):
    dev_docs = set()
    dev_sentences = load_sentences(dev_path)
    for d_sent in dev_sentences:
        sent_doc = d_sent[0][doc_idx]
        dev_docs.add(f"{sent_doc}.ann")

    return list(dev_docs)


def copy_files(source_dir, dev_path):
    data_dir = dirname(dev_path)
    dest_dir_raw = join_path(data_dir, "raw")
    create_directory(dest_dir_raw)
    docs = get_doc_annotation_files(dev_path)
    for doc_path in docs:
        source_f_path = join_path(source_dir, doc_path)
        dest_f_path = join_path(dest_dir_raw, doc_path)
        copy2(source_f_path, dest_f_path)
