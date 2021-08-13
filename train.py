import argparse
import logging
import timeit
import flair
import torch
from datetime import datetime
from torch.optim import AdamW, SGD
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.embeddings import (
    CharacterEmbeddings,
    BytePairEmbeddings,
    WordEmbeddings,
    FlairEmbeddings,
    ELMoEmbeddings,
    TransformerWordEmbeddings,
    StackedEmbeddings,
)
from flair.trainers import ModelTrainer
from torch.optim.lr_scheduler import OneCycleLR
from flair.training_utils import AnnealOnPlateau
from utils import set_logger, create_directory, join_path, seed_all
from evaluate import evaluate_meddoprof


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        dest="data_folder",
        help="path of the data dirctory",
        required=True,
    )
    parser.add_argument(
        "--hidden_size",
        dest="hidden_size",
        help="Hidden size of BiLSTM",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--rnn", dest="rnn", help="Use an rnn layer", type=bool, default=True
    )
    parser.add_argument("--crf", dest="crf", help="Use CRF", type=bool, default=True)
    parser.add_argument(
        "--reproject_embeddings",
        dest="reproject_embeddings",
        help="reproject_embeddings",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--char_emb", dest="char_emb", help="Use Character embeddings (0/1)", type=bool
    )
    parser.add_argument(
        "--byte_pair", dest="byte_pair", help="Use BytePair embeddings (en/es/multi)"
    )
    parser.add_argument(
        "--classic_emb",
        dest="classic_emb",
        help="Flair identifier for classic word embeddings",
    )
    parser.add_argument(
        "--classic_emb_1",
        dest="classic_emb_1",
        help="Flair identifier for classic word embeddings",
    )
    parser.add_argument(
        "--flair_emb",
        dest="flair_emb",
        help="Flair identifier for flair (contextualized) word embeddings",
    )
    parser.add_argument(
        "--transformer_emb",
        dest="transformer_emb",
        help="Transformer identifier for (contextualized) word embeddings",
    )
    parser.add_argument(
        "--finetune_transformer",
        dest="finetune_transformer",
        help="fine-tune transformer embeddings",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--use_scalar_mix",
        dest="use_scalar_mix",
        help="scalar mix algorithm for transformer fine-tuning",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--elmo_emb",
        dest="elmo_emb",
        help="ELMO identifier for (contextualized) word embeddings",
    )
    parser.add_argument(
        "--optimizer", dest="optimizer", help="optimizer", default="SGD"
    )
    parser.add_argument(
        "--lr", dest="lr", help="learning rate", type=float, default=0.1
    )
    parser.add_argument(
        "--one_cycle_lr",
        dest="one_cycle_lr",
        help="One Cycle R learning rate scheduler",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", help="Batch size", type=int, default=32
    )
    parser.add_argument(
        "--epochs", dest="epochs", help="Training epochs", type=int, default=150
    )
    parser.add_argument("--seed", dest="seed", help="magic seed", type=int, default=42)
    parser.add_argument(
        "--device",
        dest="device",
        help="Compute device cpu/cuda:0/cuda:1 etc.,",
        default="",
    )

    return parser


def parse_args(parser):
    args = parser.parse_args()
    return args


def get_current_time():
    return datetime.now().strftime("%d-%m-%Y__%Hh.%Mm.%Ss")


def get_data_strategy(file_path):
    data_strategy = str(file_path.split("data-strategy=")[-1])
    assert data_strategy in ["1", "2", "3", "4", "5", "6", "7", "8"]
    return data_strategy


def get_model_path(args, base_path="models/"):
    path = ""
    if args.classic_emb:
        path += "classic_emb={},".format(args.classic_emb.split("/")[-1])
    if args.classic_emb_1:
        path += "classic_emb_1={},".format(args.classic_emb_1.split("/")[-1].split(".")[0])
    if args.flair_emb:
        path += "flair={},".format(args.flair_emb)
    if args.transformer_emb:
        path += "transformer_emb={},".format(args.transformer_emb)
    if args.elmo_emb:
        path += "elmo_emb=True,"
    if args.char_emb:
        path += "char=True,"
    if args.byte_pair:
        path += "byte_pair=True,"
    path += ",{}".format(get_current_time())
    dataset_path = args.data_folder.split("/")[1]
    data_strategy = "ds={}".format(get_data_strategy(args.data_folder))
    data_path = join_path(dataset_path, data_strategy)
    path = join_path(data_path, path)
    path = join_path(base_path, path)

    return path


def train_flair(args):
    if args.device:
        flair.device = torch.device(args.device)
    if args.optimizer.lower() == "sgd":
        optimizer = SGD
    else:
        optimizer = AdamW
    if args.one_cycle_lr:
        scheduler = OneCycleLR
    else:
        scheduler = AnnealOnPlateau

    columns = {0: "text", 1: "pos", 8: "ner"}

    corpus = ColumnCorpus(
        args.data_folder,
        columns,
        train_file="train.txt",
        test_file="dev.txt",
        dev_file="dev.txt",
    )
    corpus.filter_empty_sentences()

    tag_type = "ner"
    tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)

    logging.info("tag_dict: {}".format(tag_dict))

    embedding_types = []
    if args.char_emb:
        char_embeddings = CharacterEmbeddings()
        embedding_types.append(char_embeddings)
    if args.byte_pair:
        byte_pair_embeddings = BytePairEmbeddings(args.byte_pair)
        embedding_types.append(byte_pair_embeddings)
    if args.classic_emb:
        embedding_classic = WordEmbeddings(args.classic_emb)
        embedding_types.append(embedding_classic)
    if args.classic_emb_1:
        embedding_classic_1 = WordEmbeddings(args.classic_emb_1)
        embedding_types.append(embedding_classic_1)
    if args.flair_emb:
        embeddings_flair_fw = FlairEmbeddings("{}-forward".format(args.flair_emb))
        embeddings_flair_bw = FlairEmbeddings("{}-backward".format(args.flair_emb))
        embedding_types.append(embeddings_flair_fw)
        embedding_types.append(embeddings_flair_bw)
    if args.transformer_emb:
        embeddings_transformer = TransformerWordEmbeddings(
            model=args.transformer_emb,
            fine_tune=args.finetune_transformer,
            use_scalar_mix=args.use_scalar_mix,
        )
        embedding_types.append(embeddings_transformer)
    if args.elmo_emb:
        embeddings_elmo = ELMoEmbeddings(args.elmo_emb)
        embedding_types.append(embeddings_elmo)

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(
        hidden_size=args.hidden_size,
        embeddings=embeddings,
        tag_dictionary=tag_dict,
        tag_type=tag_type,
        use_crf=args.crf,
        use_rnn=args.rnn,
        reproject_embeddings=args.reproject_embeddings,
    )

    trainer = ModelTrainer(tagger, corpus, optimizer=optimizer)

    trainer.train(
        base_path=args.model_path,
        learning_rate=args.lr,
        mini_batch_size=args.batch_size,
        max_epochs=args.epochs,
        monitor_test=True,
        scheduler=scheduler,
        patience=3,
    )


def train(args):
    start_time = timeit.default_timer()
    args.model_path = get_model_path(args)
    create_directory(args.model_path)
    set_logger(join_path(args.model_path, "train.log"))
    train_flair(args)
    # data_path, data_dir_path, model_path, task
    _task = "ner" if "task1" in args.data_folder else "class"
    _data_path = join_path(args.data_folder, "dev.txt")
    _data_dir_path = join_path(args.data_folder, "raw/")
    p, r, f1 = evaluate_meddoprof(
        data_path=_data_path,
        data_dir_path=_data_dir_path,
        model_path=args.model_path,
        task=_task,
    )
    logging.info("**meddoprof**")
    logging.info(f"p/r/f1: {p}/{r}/{f1}")
    end_time = timeit.default_timer()
    logging.info(
        "The code ran for {}m".format(round((end_time - start_time) / 60.0), 2)
    )


def main():
    parser = get_parser()
    args = parse_args(parser)
    gpu = True if torch.cuda.is_available() else False
    seed_all(args.seed, gpu)
    train(args)


if __name__ == "__main__":
    main()
