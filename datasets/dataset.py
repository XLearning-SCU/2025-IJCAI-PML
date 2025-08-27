#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from .utils import truncate_seq_pair, numpy_seed
from torchvision import transforms
import csv
import librosa

import random
class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.num_classes = len(args.labels)
        self.text_start_token = ["[CLS]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if self.args.dataset == "vsnli":
            sent1 = self.tokenizer(self.data[index]["sentence1"])
            sent2 = self.tokenizer(self.data[index]["sentence2"])
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        else:
            _ = self.tokenizer(self.data[index]["text"])
            if self.args.noise > 0.0:
                p = [0.5, 0.5]
                flag = np.random.choice([0, 1], p=p)
                if flag:
                    wordlist=self.data[index]["text"].split(' ')
                    for i in range(len(wordlist)):
                        replace_p=1/10*self.args.noise
                        replace_flag = np.random.choice([0, 1], p=[1-replace_p, replace_p])
                        if replace_flag:
                            wordlist[i]='_'
                    _=' '.join(wordlist)
                    _=self.tokenizer(_)

            sentence = (
                self.text_start_token
                + _[:(self.args.max_seq_len - 1)]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )


        if self.args.task_type == "multilabel":
            label = torch.zeros(self.num_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        if self.data[index]["img"]:
            img_path = os.path.join(self.data_dir, self.data[index]["img"])
            if self.args.dataset == "vsnli":
                img_path = os.path.join(self.data_dir,'flickr30k-images', self.data[index]["img"])
            image = Image.open(
                img_path
            ).convert("RGB")
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        image = self.transforms(image)

        return sentence, segment, image, label, torch.LongTensor([index])

class AddGaussianNoise(object):

    '''
    mean:均值
    variance：方差
    amplitude：幅值
    '''
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class AddSaltPepperNoise(object):

    def __init__(self, density=0,p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            mask = np.repeat(mask, c, axis=2)
            img[mask == 0] = 0
            img[mask == 1] = 255
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img

class AVDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = args.data_path
        self.dataset =  args.dataset

        if args.dataset == 'CREMA-D':
            self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
            self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')
 
            labels = []
            with open(self.test_csv, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)  
                for item in csv_reader:
                    if item[1] not in labels:
                        labels.append(item[1])

            class_dict = dict()
            for i,v in enumerate(labels):
                class_dict[v] = i  
            self.num_classes = len(labels)            

            if mode == 'train':
                csv_file = self.train_csv
            else:
                csv_file = self.test_csv
    
            with open(csv_file, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)  
                for item in csv_reader:
                    audio_path = os.path.join(self.data_root, args.dataset,'AudioWAV', item[0] + '.wav') 
                    visual_path = os.path.join(self.data_root, args.dataset, 'Image-01-FPS', item[0]) 
                    if os.path.exists(audio_path) and os.path.exists(visual_path):  
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[1]])
                    else:
                        continue
                    
        elif args.dataset == 'avsbench':
            self.full_csv = os.path.join(self.data_root, args.dataset,  's4_meta_data.csv')
            labels = []
            with open(self.full_csv, 'r') as f:
                files = f.readlines()
            for item in files[1:]:
                item = item.split(',')
                item[3] = item[3].replace('\n','')
                if item[2] not in labels:
                    labels.append(item[2])
            class_dict = dict()
            for i,v in enumerate(labels):
                class_dict[v] = i  
            self.num_classes = len(labels)   

 
            for class_ in labels: 
                base_path = os.path.join(self.data_root, args.dataset, f's4_data/audio_wav/{mode}/{class_}')   
                av_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
                for file in av_files: 
                    inst_name = file.replace('.wav','') 
                    audio_path = os.path.join(self.data_root, args.dataset, f's4_data/audio_wav/{mode}/{class_}/{inst_name}.wav')  
                    visual_path = os.path.join(self.data_root, args.dataset, f's4_data/visual_frames/{mode}/{class_}/{inst_name}')  
                    if os.path.exists(audio_path) and os.path.exists(visual_path): 
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[class_])
                    else: 
                        continue
        
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        print(f'{mode} samples==============>{len(self.image)} {self.num_classes}')
      

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio 
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7) 
 
 
        image_samples = os.listdir(self.image[idx])
        file_num = len(image_samples)
        pick_num = 1 
        seg = file_num//pick_num 
        i = 0
        if self.mode == 'train':
            index = random.randint(i*seg + 1, i*seg + seg)
        else:
            index = i*seg + seg//2
        img = Image.open(os.path.join(self.image[idx], image_samples[index-1])).convert('RGB')
        images = self.transform(img)
 
        label = self.label[idx]
        return spectrogram, images, label, idx 
        
 