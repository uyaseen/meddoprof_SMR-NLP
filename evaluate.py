import argparse
import codecs
import flair
import torch
from flair.models import SequenceTagger
from flair.data import Sentence
from eval_utils.meddoprof_evaluate_ner import main as evaluate_ner
from utils import create_directory, join_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        dest="data_path",
        help="path of the data file to evaluate",
        required=True,
    )
    parser.add_argument(
        "--gt_data_dir",
        dest="gt_data_dir",
        help="path of the ground truth raw annotation files",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        help="path of the trained (flair) model directory",
        required=True,
    )
    parser.add_argument(
        "--task",
        dest="task",
        help="task to evaluate on [ner, class, norm]",
        required=True,
    )
    parser.add_argument(
        "--device",
        dest="device",
        help="compute device[cpu, cuda:0]"
    )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    return args


def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, "r", "utf8"):
        line = line.rstrip()
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


def custom_to_tagged_string(sentence, only_return_predictions=True):
    tokens, tags = [], []
    for token in sentence.tokens:
        tokens.append(token.text)
        tag = "O"
        for label_type in token.annotation_layers.keys():
            if (
                token.get_labels(label_type)[0].value == "O"
                or token.get_labels(label_type)[0].value == "_"
            ):
                continue
            tag = token.get_labels(label_type)[0].value
        tags.append(tag)
    assert len(tokens) == len(tags)

    if only_return_predictions:
        return tags

    return tokens, tags


def extract_entities(words, tags, start, end, majority_selection=False):
    entities = []
    phrase = ""
    last_tag = "O"
    start_chunk = False
    end_chunk = False
    iter_idx = 0
    last_end_idx = None
    ph_st_idx = None
    ph_end_idx = None
    entity_labels = []
    for idx in range(len(words)):
        word, tag, st_idx, end_idx = words[idx], tags[idx], start[idx], end[idx]
        if last_tag == "O" and tag.startswith("B"):
            start_chunk = True
            end_chunk = False
            ph_st_idx = st_idx

        if last_tag.startswith("B") and tag == "O":
            end_chunk = True
            start_chunk = False
            ph_end_idx = last_end_idx

        if last_tag.startswith("I") and tag == "O":
            end_chunk = True
            start_chunk = False
            ph_end_idx = last_end_idx

        if (last_tag.startswith("B") and tag.startswith("I")) or (
            last_tag.startswith("I") and tag.startswith("I")
        ):
            end_chunk = False
            start_chunk = False

        if start_chunk:
            phrase = word
            start_chunk = False
            last_tag = tag
            entity_labels.append(tag.replace("B-", "").replace("I-", ""))
            if iter_idx == len(tags) - 1:
                end_chunk = False
                start_chunk = False
                # TODO: over-write everything under if with
                # end_chunk = True
                # ph_st_idx = st_idx
                # ph_end_idx = end_idx
                # entity_labels = []

        elif end_chunk:
            assert ph_st_idx is not None, "ph_st_idx == None"
            assert ph_end_idx is not None, "ph_end_idx == None"
            if majority_selection:
                _class = max(entity_labels, key=entity_labels.count)
            else:
                _class = entity_labels[0]
            entities.append(
                {
                    "text": phrase,
                    "class": last_tag.split("-")[-1],
                    "start": ph_st_idx,
                    "end": ph_end_idx,
                }
            )
            last_tag = tag
            end_chunk = False
            entity_labels = []
        elif tag == "O" and last_tag == "O":
            # outside of the phrase or chunk
            last_tag = tag
        else:
            # middle of the chunk
            phrase = phrase + " " + word
            entity_labels.append(tag.replace("B-", "").replace("I-", ""))
            last_tag = tag
            if iter_idx == len(tags) - 1:
                end_chunk = False
                start_chunk = False
        iter_idx += 1
        last_end_idx = end_idx

    return entities


def write_prediction_files(w_path, documents_dict):
    create_directory(w_path)
    entities_count = 0
    for doc_name, doc_entities in documents_dict.items():
        with codecs.open(
            join_path(w_path, "{}.ann".format(doc_name)), "w", encoding="utf-8"
        ) as f:
            for e_idx, ent in enumerate(doc_entities):
                ent_str = "T{}\t{} {} {}\t{}".format(
                    e_idx + 1, ent["class"], ent["start"], ent["end"], ent["text"]
                )
                f.write("{}\n".format(ent_str))
                entities_count += 1

    print("{} annotation files created".format(len(documents_dict)))
    print("entities detected: {}".format(entities_count))


def evaluate_meddoprof(
    data_path, data_dir_path, model_path, task, dir_name="dev", batch_size=32
):
    data_sentences = load_sentences(data_path)
    tagger = SequenceTagger.load(join_path(model_path, "best-model.pt"))

    predictions_steps = int(len(data_sentences) / 32)
    print("Number of batches: {}".format(predictions_steps))

    documents_dict = {}

    for i in range(predictions_steps):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_sentences = data_sentences[start:end]
        flair_sentences = []
        sentences_tokens = []
        sentences_docs = []
        sentences_starts = []
        sentences_ends = []
        sentences_tags = []
        for b_sentence in batch_sentences:
            tokens = [s[0] for s in b_sentence]
            sentences_tokens.append(tokens)
            _sentence = Sentence([s[0] for s in b_sentence], use_tokenizer=False)
            flair_sentences.append(_sentence)
            _sentence_docs = [s[2] for s in b_sentence]
            _sentence_starts = [s[3] for s in b_sentence]
            _sentence_ends = [s[4] for s in b_sentence]
            _sentence_tags = [s[-1] for s in b_sentence]
            sentences_docs.append(_sentence_docs)
            sentences_starts.append(_sentence_starts)
            sentences_ends.append(_sentence_ends)
            sentences_tags.append(_sentence_tags)
        tagger.predict(flair_sentences)
        predictions = []
        predictions_tokens = []
        for sentence in flair_sentences:
            tkn, tgs = custom_to_tagged_string(sentence, only_return_predictions=False)
            predictions.append(tgs)
            predictions_tokens.append(tkn)
        # predictions = [custom_to_tagged_string(sentence) for sentence in flair_sentences]
        for j in range(batch_size):
            sent_tokens = sentences_tokens[j]
            sent_docs = sentences_docs[j]
            sent_token_starts = sentences_starts[j]
            sent_token_ends = sentences_ends[j]
            sent_tags = sentences_tags[j]
            sent_preds = predictions[j]
            assert (
                len(sent_tokens)
                == len(sent_docs)
                == len(sent_token_starts)
                == len(sent_token_ends)
                == len(sent_tags)
                == len(sent_preds)
            ), (
                "tokens:{}, docs: {}, "
                "tags:{},preds:{}, "
                "token_starts:{},token_ends:{}".format(
                    len(sent_tokens),
                    len(sent_docs),
                    len(sent_token_starts),
                    len(sent_token_ends),
                    len(sent_tags),
                    len(sent_preds),
                )
            )

            # fill-in documents_dict
            sent_entities = extract_entities(
                sent_tokens, sent_preds, sent_token_starts, sent_token_ends
            )
            doc_idx = sent_docs[0]
            if doc_idx not in documents_dict:
                documents_dict[doc_idx] = []
            if len(sent_entities) > 0:
                documents_dict[doc_idx] += sent_entities

    # meddoprof evaluate
    pred_path = join_path(model_path, dir_name)
    print("pred_path {}".format(pred_path))
    print("gt_path {}".format(data_dir_path))
    write_prediction_files(pred_path, documents_dict)
    p, r, f1 = evaluate_ner(data_dir_path, pred_path, subtask=task)
    p, r, f1 = round(p, 4), round(r, 4), round(f1, 4)

    return p, r, f1


def main():
    parser = get_parser()
    args = parse_args(parser)
    if args.device:
        flair.device = torch.device(args.device)
    p, r, f1 = evaluate_meddoprof(
        args.data_path, args.gt_data_dir, args.model_path, args.task
    )
    print(f"*meddoprof - {args.task}*")
    print(f"P: {p}, R: {r}, F1: {f1}")


if __name__ == "__main__":
    main()
