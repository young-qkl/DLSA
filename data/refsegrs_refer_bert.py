import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from bert.tokenization_bert import BertTokenizer


def build_refsegrs_batches(split, data_root):
    image_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')

    if split == 'train':
        txt_file = 'output_phrase_train.txt'
    elif split == 'val':
        txt_file = 'output_phrase_val.txt'
    elif split == 'test':
        txt_file = 'output_phrase_test.txt'
    else:
        raise ValueError(f'Unsupported split: {split}')

    txt_path = os.path.join(data_root, txt_file)

    all_imgs = []
    all_masks = []
    all_sentences = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        img_name = parts[0]
        sentence = ' '.join(parts[1:]).strip()

        img_path = os.path.join(image_dir, img_name + '.tif')
        mask_path = os.path.join(mask_dir, img_name + '.tif')

        all_imgs.append(img_path)
        all_masks.append(mask_path)
        all_sentences.append(sentence)

    print(f"RefSegRS loaded from: {data_root}, split={split}, samples={len(all_imgs)}")
    return all_imgs, all_masks, all_sentences


class ReferDataset(data.Dataset):
    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.eval_mode = eval_mode

        self.data_root = args.refsegrs_data_root
        self.max_tokens = 20
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.imgs, self.labels, self.sentences = build_refsegrs_batches(
            self.split, self.data_root
        )

        self.input_ids = []
        self.attention_masks = []

        # 按你原来简化版 dataset_refer_bert 的风格来做
        for sentence_raw in self.sentences:
            attention_mask = [0] * self.max_tokens
            padded_input_ids = [0] * self.max_tokens

            input_ids = self.tokenizer.encode(
                text=sentence_raw,
                add_special_tokens=True
            )

            input_ids = input_ids[:self.max_tokens]
            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1] * len(input_ids)

            # 保持成“每个样本一个 list”的形式，兼容你原来的写法
            self.input_ids.append([torch.tensor(padded_input_ids).unsqueeze(0)])
            self.attention_masks.append([torch.tensor(attention_mask).unsqueeze(0)])

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        mask_path = self.labels[index]

        img = Image.open(img_path).convert("RGB")

        label_mask = cv2.imread(mask_path, 0)
        if label_mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # 按你原文件的逻辑，阈值化成二值 mask
        ref_mask = np.array(label_mask) > 50
        annot = np.zeros(ref_mask.shape, dtype=np.uint8)
        annot[ref_mask == 1] = 1
        annot = Image.fromarray(annot, mode="P")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)
        else:
            target = annot

        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask