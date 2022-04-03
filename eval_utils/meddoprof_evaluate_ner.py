#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:32:08 2021

@author: tonifuc3m
"""
import pandas as pd
import os

from eval_utils.ann_parsing import main as ann_main
from eval_utils.compute_metrics import main as comp_main


def main(
    gs_path,
    pred_path,
    subtask=["class", "ner", "norm"],
    codes_path="meddoprof_valid_codes",
):
    """
    Load GS and Predictions; format them; compute precision, recall and 
    F1-score and print them.

    Parameters
    ----------
    gs_path : str
        Path to directory with GS in Brat (or to GS in TSV if subtask is norm).
    pred_path : str
        Path to directory with Predicted files in Brat (or to GS in TSV if subtask is norm).
    subtask : str
        Subtask name
    codes_path : str
        Path to TSV with valid codes

    Returns
    -------
    None.

    """

    if subtask == "norm":
        gs = pd.read_csv(gs_path, sep="\t", header=0)
        pred = pd.read_csv(pred_path, sep="\t", header=0)

        if pred.shape[0] == 0:
            raise Exception("There are not parsed predicted annotations")
        elif gs.shape[0] == 0:
            raise Exception("There are not parsed Gold Standard annotations")
        if pred.shape[1] != 4:
            raise Exception("Wrong column number in predictions file")
        elif gs.shape[1] != 4:
            raise Exception("Wrong column number in Gold Standard file")

        gs.columns = ["clinical_case", "span", "offset", "code"]
        pred.columns = ["clinical_case", "span", "offset", "code"]

        pred["offset"] = pred["offset"].apply(lambda x: x.strip())
        pred["code"] = pred["code"].apply(lambda x: x.strip())
        pred["clinical_case"] = pred["clinical_case"].apply(lambda x: x.strip())

    elif subtask in ["class", "ner"]:

        if subtask == "class":
            labels = ["SANITARIO", "PACIENTE", "FAMILIAR", "OTROS"]
        elif subtask == "ner":
            labels = ["ACTIVIDAD", "PROFESION", "SITUACION_LABORAL"]

        gs = ann_main(gs_path, labels, with_notes=False)
        pred = ann_main(pred_path, labels, with_notes=False)

        if pred.shape[0] == 0:
            raise Exception("There are not parsed predicted annotations")
        elif gs.shape[0] == 0:
            raise Exception("There are not parsed Gold Standard annotations")

        gs.columns = [
            "clinical_case",
            "mark",
            "label",
            "offset",
            "span",
            "start_pos_gs",
            "end_pos_gs",
        ]
        pred.columns = [
            "clinical_case",
            "mark",
            "label",
            "offset",
            "span",
            "start_pos_pred",
            "end_pos_pred",
        ]

    # Drop duplicates
    pred = pred.drop_duplicates().copy()
    gs = gs.drop_duplicates().copy()

    # Remove predictions for files not in Gold Standard
    if subtask in ["ner", "class"]:
        doc_list_gs = list(filter(lambda x: x[-4:] == ".ann", os.listdir(gs_path)))
    elif subtask == "norm":
        doc_list_gs = list(set(gs["clinical_case"].tolist()))
    pred_gs_subset = pred.loc[pred["clinical_case"].isin(doc_list_gs), :].copy()

    if pred_gs_subset.shape[0] == 0:
        raise Exception(
            "There are not valid predicted annotations. "
            + "The only predictions are for files not in Gold Standard"
        )

    # Remove predictions for codes not valid
    if subtask == "norm":
        valid_codes = pd.read_csv(codes_path, sep="\t", header=0)["code"].tolist()
        pred_gs_subset = pred_gs_subset.loc[pred["code"].isin(valid_codes), :].copy()

    if pred_gs_subset.shape[0] == 0:
        raise Exception(
            "There are not valid predicted annotations. "
            + "The only predictions contain invalid codes"
        )

    # Compute metrics
    P_per_cc, P, R_per_cc, R, F1_per_cc, F1 = comp_main(
        gs, pred_gs_subset, doc_list_gs, subtask=subtask
    )

    return P, R, F1
