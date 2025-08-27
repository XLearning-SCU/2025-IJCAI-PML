#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from .dataset import JsonlDataset, AddGaussianNoise, AddSaltPepperNoise, AVDataset
from .vocab import Vocab


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    
def get_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

def get_GaussianNoisetransforms(rgb_severity):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply([AddGaussianNoise(amplitude=rgb_severity * 10)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

def get_SaltNoisetransforms(rgb_severity):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply([AddSaltPepperNoise(density=0.1, p=rgb_severity/10)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
def get_labels_and_frequencies(path):
    label_freqs = Counter()
    # print(path)
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    idx=torch.cat([row[4] for row in batch]).long()
    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor,idx


def get_data_loaders(args):
    tokenizer = (
        # BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    )

    transforms = get_transforms()

    train_transforms = get_train_transforms()
    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.dataset, "train.jsonl")
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)
    train = JsonlDataset(
        os.path.join(args.data_path, args.dataset, "train.jsonl"),
        tokenizer,
        # train_transforms,
        transforms,
        vocab,
        args,
    )
    args.train_data_len = len(train)

    dev = JsonlDataset(
        os.path.join(args.data_path, args.dataset, "dev.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    if args.noise>0.0:
        if args.noise_type=='Gaussian':
            print('Gaussian')
            test_transforms=get_GaussianNoisetransforms(args.noise)
        elif args.noise_type=='Salt':
            print("Salt")
            test_transforms = get_SaltNoisetransforms(args.noise)
    else:
        test_transforms=transforms


    test_set = JsonlDataset(
        os.path.join(args.data_path, args.dataset, "test.jsonl"),
        tokenizer,
        test_transforms,
        vocab,
        args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    if args.dataset == "vsnli":
        test_hard = JsonlDataset(
            os.path.join(args.data_path, args.dataset, "test_hard.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        test_hard_loader = DataLoader(
            test_hard,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        test = {"test": test_loader, "test_hard": test_hard_loader}
    elif args.dataset == "MVSA_Single":
        test = {"test": test_loader}

    elif args.dataset == "food101":
        test = {"test": test_loader}
    else:
        test_gt = JsonlDataset(
            os.path.join(args.data_path, args.dataset, "test_hard_gt.jsonl"),
            tokenizer,
            test_transforms,
            vocab,
            args,
        )

        test_gt_loader = DataLoader(
            test_gt,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )


        test = {
            "test": test_loader,
            "test_gt": test_gt_loader,
        }

    return train_loader, val_loader, test


def get_va_data_loaders(args):
    train_dataset = AVDataset(args, mode='train')
    test_dataset = AVDataset(args, mode='test')
    try:
        val_dataset = AVDataset(args, mode='val')
    except:
        print("===")
        val_dataset = test_dataset 
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=False)
    
    return train_loader, val_loader, test_loader
    