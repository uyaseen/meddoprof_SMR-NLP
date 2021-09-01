# meddoprof_SMR-NLP

This repostiroy contains the source code and models of our submission (team: SMR-NLP) in the [MEDDOPROF](https://temu.bsc.es/meddoprof/) shared task.

# Setting-up

## Data

Download the [train, development](https://zenodo.org/record/4775741) and [test](https://zenodo.org/record/5077976) set for MEDDOPROF shared task.

Extract all the data in the `data/` directory and run `python data.py`.

## Embeddings

Download the [fastText Spanish medical embeddings](https://zenodo.org/record/3744326).

## Dependencies

Download the fork of `flair` from [here](https://drive.google.com/drive/folders/1QThWNAQyiKvo1hPVECnZIa6oe8KSA3D-?usp=sharing), install it with the command:

```python
pip install .
```

Install the required pacakges using the command:

```python
pip install -r requirements.txt
```

# Training

The best combination of embeddings were based on our initial experiments which are documented [here](https://docs.google.com/spreadsheets/d/1NiZWRSXM8JJYgB9sA0QwABSHiHKYocPqh4ivw8p_v0w/edit?usp=sharing).

## Task 1

```python
python train.py --data_folder data/task1/data-strategy=1 --classic_emb PATH_TO_EMBEDDINGS/biomedical_embeddings_for_spanish_v2.0/scielo_wikipedia/skipgram/cased/Scielo+Wiki_skipgram_cased.vec --transformer_emb dccuchile/bert-base-spanish-wwm-cased --device cuda:0
```

## Task 2

```python
python train.py --data_folder data/task2/data-strategy=1 --classic_emb es --transformer_emb dccuchile/bert-base-spanish-wwm-cased --byte_pair es --device cuda:0
```

# Pre-trained Models

Pre-trained models for task 1 and task 2 are available [here](https://drive.google.com/drive/folders/1xvO7wriln34WAE50jiRcqfrmgPqZ4Vh5).

# Evaluation

## Task 1

```python
python evaluate.py  --data_path data/task1/test.txt --gt_data_dir data/meddoprof-test-GS/class/ --model_path models/task1/ds=1/bert-base-spanish-wwm-cased,,02-06-2021__07h.46m.10s/ --task ner --device cuda:0
```

```
*meddoprof - ner*
P: 0.854, R: 0.751, F1: 0.799
```

## Task 2

```python
python evaluate.py  --data_path data/task2/test.txt --gt_data_dir data/meddoprof-test-GS/class/ --model_path models/task2/ds=1/bert-base-spanish-wwm-cased,byte_pair=True,,17-05-2021__08h.59m.58s/ --task class --device cuda:0
```

```
*meddoprof - class*
P: 0.8021, R: 0.6986, F1: 0.7468
```


# 📕 Citation

This codebase was originally developed as part of our submission in the SMM4H shared task. If you make use of the code in this repository, please cite our paper:

```
@inproceedings{yaseen-langer-2021-neural,
    title = "Neural Text Classification and Stacked Heterogeneous Embeddings for Named Entity Recognition in {SMM}4{H} 2021",
    author = "Yaseen, Usama  and
      Langer, Stefan",
    booktitle = "Proceedings of the Sixth Social Media Mining for Health ({\#}SMM4H) Workshop and Shared Task",
    month = jun,
    year = "2021",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.smm4h-1.14",
    doi = "10.18653/v1/2021.smm4h-1.14",
    pages = "83--87"
}
```
