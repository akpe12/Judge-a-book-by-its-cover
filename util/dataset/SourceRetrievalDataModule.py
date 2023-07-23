import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import torch

from transformers import AutoTokenizer
from tqdm import tqdm
# import pyarrow as pa

from PIL import Image, ImageFile
from io import BytesIO
import  base64
from urllib import request

import json
import random 

from torchvision import transforms
import json, random, time, os, base64
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from BasicDataset import BasicDataset
from BasicDatasetForBook import BasicDatasetForBook
from BasicDatasetForBookTest import BasicDatasetForBookTest

class SourceRetrievalDataModule(LightningDataModule):
    def __init__(self, _config, dist=True):
        super().__init__()
        self._config = _config
        self.per_gpu_batch_size = _config['per_gpu_batch_size']
        self.num_workers = _config['num_workers']
        self.input_seq_len = _config['input_seq_len']
        self.dist = dist
        
        self.train_dataset_path = _config['train_dataset_path']
        self.val_dataset_path = _config['val_dataset_path']
        self.test_dataset_path = _config['test_dataset_path']
        self.model_name = _config['model_name']
        
    def setup(self, stage=None):
        if self._config['mode'] == 'test':
            self.test_dataset = LoadDataset(self._config, self.model_name, self.test_dataset_path, seq_len=self.input_seq_len, mode=self._config['mode'])
            
            # self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_dataset = LoadDataset(self._config, self.model_name, self.train_dataset_path,  seq_len=self.input_seq_len, mode=self._config['mode'])        
            self.val_dataset = LoadDataset(self._config, self.model_name, self.val_dataset_path, seq_len=self.input_seq_len, mode=self._config['mode'])
            
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        
    def traindataset_reinit_at_train_epoch_end(self, current_epoch_plus):
        self.train_dataset = LoadDataset(self._config, self.model_name, self.train_dataset_path, seq_len=self.input_seq_len, mode=self._config['mode']) 
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.per_gpu_batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.per_gpu_batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.per_gpu_batch_size,
            # sampler=self.test_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    
    
class LoadDataset(Dataset):
    def __init__(self, _config, model_name, corpus_path, seq_len, mode):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        # self.tokenizer = OFATokenizer.from_pretrained(model_name, sep_token="<sep>", cls_token="<cls>")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        if mode == "train":
            data = BasicDatasetForBook.load(self.corpus_path)
            self._data = self.make_dataset(data, _config)
        else:
            data = BasicDatasetForBookTest.load(self.corpus_path)
            self._data = self.make_dataset_test(data, _config)
        
    def make_dataset(self, data, _config):
        data = list(data)
        processed_data = []
        # image_index = []
        
        for i in range(len(data)):
            title = data[i]['title']
            label = data[i]['label']
            # image_index.append(i)
            
            # text = self.tokenizer.encode(title, add_special_tokens=False)
            
            # # padding
            # if len(text) <= self.seq_len - 2:
                # text = [self.cls_token_id] + text + [self.sep_token_id]
            #     pad_length = self.seq_len - len(text)
            #     attention_mask = (len(text) * [1]) + (pad_length * [0])
            #     text = text + (pad_length * [self.pad_token_id])
            # # special token 넣을 자리가 없는 seq라면
            # else:
            #     text = text[:self.seq_len - 2]
            #     text = [self.cls_token_id] + text + [self.sep_token_id]
            #     attention_mask = len(text) * [1]
            
            text = self.tokenizer.encode_plus(title, add_special_tokens=True, padding='max_length', max_length= _config['input_seq_len'], return_attention_mask=True)
            
            # input_ids = text
            
            # img_path = "/home/yangcw/mm_cls/1_source_retrieval/Image/Image/" + str(i) + ".jpg"
            
            processed_data.append({"input_ids":text['input_ids'],
                                    "attention_mask":text['attention_mask'],
                                    "labels":label})
            
        return processed_data
    
    def make_dataset_test(self, data, _config):
        data = list(data)
        processed_data = []
        
        for i in range(len(data)):
            title = data[i]['title']
    
            text = self.tokenizer.encode_plus(title, add_special_tokens=True, padding='max_length', max_length= _config['input_seq_len'], return_attention_mask=True)
            
            processed_data.append({"input_ids":text['input_ids'],
                                    "attention_mask":text['attention_mask']})
            
        return processed_data
    
    # 여기서 이미지
    def __getitem__(self, index) -> dict:
        input = self._data[index]
        
        input_dict = {}
        input_dict.update(input)
        
        return {k:torch.tensor(v) for k, v in input_dict.items()}
    
    def __len__(self):
        return len(self._data)
        

        # # image loading
        # mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        # self.resolution = _config['image_resolution']
        # self.patch_resize_transform = transforms.Compose([
        #     lambda image: image.convert("RGB"),
        #     transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
        #     transforms.ToTensor(), 
        #     transforms.Normalize(mean=mean, std=std)
        # ])
        
        # # Load Image arrow table
        # self.table = pa.ipc.RecordBatchFileReader(
        #         pa.memory_map(self.image_arrow_path, "r")
        #     ).read_all()
        # idlist = self.table['image_id']
        # self.imageid_to_arrowid = {}        
        # for arrow_id, image_id in enumerate(idlist):
        #     self.imageid_to_arrowid[image_id.as_py()] = arrow_id     
        # # print(self.imageid_to_arrowid[30278219])      
        # # # print(self.table['image'][self.imageid_to_arrowid[30278219]])
        # # print(Image.open(BytesIO(base64.b64decode(self.table['image'][self.imageid_to_arrowid[30278219]].as_py()))))

        # with open(corpus_path) as f:
        #     dataset = json.load(f)
            
        #     dataset_li = []
            
        #     labels = []
            
        #     img_ids = []
        #     txt_ids = []
        #     id_to_utterences = {}
        #     for data_id in dataset.keys():
        #         for posfact in dataset[data_id]['img_posFacts']:
        #             img_ids.append(posfact['image_id'])
        #             id_to_utterences[posfact['image_id']] = posfact
        #         for negfact in dataset[data_id]['img_negFacts']:
        #             img_ids.append(negfact['image_id'])
        #             id_to_utterences[negfact['image_id']] = negfact
        #         for posfact in dataset[data_id]['txt_posFacts']:
        #             txt_ids.append(posfact['snippet_id'])
        #             id_to_utterences[posfact['snippet_id']] = posfact
        #         for negfact in dataset[data_id]['txt_negFacts']:
        #             txt_ids.append(negfact['snippet_id'])
        #             id_to_utterences[negfact['snippet_id']] = negfact


        #     for data_id in dataset.keys():
        #         # if dataset[data_id]['split'] == 'train':
        #         #     random.shuffle(dataset[data_id]['img_negFacts'])
        #         #     random.shuffle(dataset[data_id]['txt_negFacts'])

                       
        #         if len(dataset[data_id]['img_posFacts']) > 0 and dataset[data_id]['split'] == mode:
        #             question = dataset[data_id]['Q']
                    
        #             posfacts_ids = []
        #             negfacts_ids = []
                    
        #             for posfact in dataset[data_id]['img_posFacts']:
        #                 dataset_li.append({'img_id': posfact['image_id'], 'Q': question, 'title': posfact['title'], 'caption': posfact['caption'], 'label': 1})
        #                 labels.append(1)
                        
        #                 posfacts_ids.append(posfact['image_id'])
                    
                    
        #             # random.shuffle(img_ids)
        #             # random.shuffle(txt_ids) 
                    
        #             for img_id in random.sample(img_ids,4): # neg fact
        #                 if img_id not in posfacts_ids:
        #                     dataset_li.append({'img_id': img_id, 'Q': question, 
        #                                     'title': id_to_utterences[img_id]['title'], 'caption': id_to_utterences[img_id]['caption'], 'label': 0})
        #                     labels.append(0)
        #                     negfacts_ids.append(img_id)
                        
        #                 if len(negfacts_ids) == 2:
        #                     break
                        
        #             for txt_id in random.sample(txt_ids,4): # neg fact
        #                 if txt_id not in posfacts_ids:
        #                     dataset_li.append({'img_id': None, 'Q': question, 
        #                                     'title': id_to_utterences[txt_id]['title'], 'caption': id_to_utterences[txt_id]['fact'], 'label': 0})                    
        #                     labels.append(0)
        #                     negfacts_ids.append(txt_id)
        #                 if len(negfacts_ids) == 3:
        #                     break                        
                        
        #             # for posfact in dataset[data_id]['img_negFacts'][:2]:
        #             #     dataset_li.append({'img_id': posfact['image_id'], 'Q': question, 'title': posfact['title'], 'caption': posfact['caption'], 'label': 0})
        #             #     labels.append(0)

        #             # for posfact in dataset[data_id]['txt_negFacts'][:1]:
        #             #     dataset_li.append({'img_id': None, 'Q': question, 'title': posfact['title'], 'caption': posfact['fact'], 'label': 0})
        #             #     labels.append(0)

        #             neg_img_num = 0
        #             neg_txt_num = 0
        #             for retrieved_fact in dataset[data_id]['retrieved_facts']:
        #                 if 'image_id' in retrieved_fact.keys():
        #                     if retrieved_fact['image_id'] not in posfacts_ids and neg_img_num < 4:
        #                         dataset_li.append({'img_id': None, 'Q': question, 'title': retrieved_fact['title'], 'caption': retrieved_fact['caption'], 'label': 0})
        #                         labels.append(0)
                                
        #                         negfacts_ids.append(retrieved_fact['image_id'])
        #                         neg_img_num += 1
                                
        #                 if 'snippet_id' in retrieved_fact.keys():
        #                     if retrieved_fact['snippet_id'] not in posfacts_ids and neg_txt_num < 1:
        #                         dataset_li.append({'img_id': None, 'Q': question, 'title': retrieved_fact['title'], 'caption': retrieved_fact['fact'], 'label': 0})
        #                         labels.append(0)
                                
        #                         negfacts_ids.append(retrieved_fact['snippet_id'])
        #                         neg_txt_num += 1    
        #             # print('image')                            
        #             # print(posfacts_ids)
        #             # print(negfacts_ids)                        
                        
        #         elif len(dataset[data_id]['txt_posFacts']) > 0 and dataset[data_id]['split'] == mode:
        #             question = dataset[data_id]['Q']

        #             posfacts_ids = []
        #             negfacts_ids = []
                                        
        #             for posfact in dataset[data_id]['txt_posFacts']:
        #                 dataset_li.append({'img_id': None, 'Q': question, 'title': posfact['title'], 'caption': posfact['fact'], 'label': 1})
        #                 labels.append(1)
                        
        #                 posfacts_ids.append(posfact['snippet_id'])

        #             # random.shuffle(img_ids)
        #             # random.shuffle(txt_ids) 
        #             for txt_id in random.sample(txt_ids,4): # neg fact
        #                 if txt_id not in posfacts_ids:
        #                     dataset_li.append({'img_id': None, 'Q': question, 
        #                                        'title': id_to_utterences[txt_id]['title'], 'caption': id_to_utterences[txt_id]['fact'], 'label': 0})                    
        #                     labels.append(0)
        #                     negfacts_ids.append(txt_id)
        #                 if len(negfacts_ids) == 3:
        #                     break      
                        
        #             for img_id in random.sample(img_ids,4): # neg fact
        #                 if img_id not in posfacts_ids:
        #                     dataset_li.append({'img_id': img_id, 'Q': question, 
        #                                     'title': id_to_utterences[img_id]['title'], 'caption': id_to_utterences[img_id]['caption'], 'label': 0})
        #                     labels.append(0)
        #                     negfacts_ids.append(img_id)
                        
        #                 if len(negfacts_ids) == 4:
        #                     break   
                                                                                             
        #             # for posfact in dataset[data_id]['txt_negFacts'][:3]:
        #             #     dataset_li.append({'img_id': None, 'Q': question, 'title': posfact['title'], 'caption': posfact['fact'], 'label': 0})
        #             #     labels.append(0)
        #             #     # break 
        #             # for posfact in dataset[data_id]['img_negFacts'][:1]:
        #             #     dataset_li.append({'img_id': posfact['image_id'], 'Q': question, 'title': posfact['title'], 'caption': posfact['caption'], 'label': 0})
        #             #     labels.append(0)  
                        
                        
        #             neg_img_num = 0                        
        #             neg_txt_num = 0
        #             for retrieved_fact in dataset[data_id]['retrieved_facts']:
        #                 if 'snippet_id' in retrieved_fact.keys():
        #                     if retrieved_fact['snippet_id'] not in posfacts_ids and neg_txt_num < 4:
        #                         dataset_li.append({'img_id': None, 'Q': question, 'title': retrieved_fact['title'], 'caption': retrieved_fact['fact'], 'label': 0})
        #                         labels.append(0)
                                
        #                         negfacts_ids.append(retrieved_fact['snippet_id'])
        #                         neg_txt_num += 1
                                
        #                 if 'image_id' in retrieved_fact.keys():
        #                     if retrieved_fact['image_id'] not in posfacts_ids and neg_img_num < 1:
        #                         dataset_li.append({'img_id': None, 'Q': question, 'title': retrieved_fact['title'], 'caption': retrieved_fact['caption'], 'label': 0})
        #                         labels.append(0)
                                
        #                         negfacts_ids.append(retrieved_fact['image_id'])
        #                         neg_img_num += 1
                    # print('text')                                                            
                    # print(posfacts_ids)
                    # print(negfacts_ids)
            
        # # for class weighting             
        # labels = torch.tensor(labels)
        # assert labels.dim() == 1
        # classes, samples_per_class = torch.unique(labels, return_counts=True)
        # assert len(classes) == 2
        # weights = len(labels) / (len(classes) * samples_per_class.float())
        # class_weights = weights / weights.sum()
        # print(class_weights)       # train : 0-260488 1-36226
            
    #     self.processed_dataset = []

        
    #     for data in tqdm(dataset_li[:]):
    #         text = data['title']
    #         label = data['label']
            
    #         text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    #         if len(text) <= self.seq_len - 2:
    #             text = [self.start] + text + [self.sep]
    #             pad_length = self.seq_len - len(text)

    #             attention_mask = (len(text) * [1]) + (pad_length * [0])
    #             text = text + (pad_length * [self.padding])
    #         else:
    #             text = text[:self.seq_len - 2] # 그냥 조금 자르고 special token 붙임
    #             text = [self.start] + text + [self.sep]
    #             attention_mask = len(text) * [1]

    #         model_input = text
    #         self.processed_dataset.append({"input_ids": model_input,
    #                                        'patch_images': data['img_id'],  # 임시로 id값 저장 후 getitem에서 image 처리.
    #                                        'patch_masks': [True]  * (self.resolution // 16) * (self.resolution // 16), 
    #                                        "labels": label,
    #                                        'decoder_input_ids': [self.sep] })            


    # def __len__(self):
    #     return len(self.processed_dataset)

    # def __getitem__(self, item):
    #     output = self.processed_dataset[item]
        
    #     item_dict = {}
    #     item_dict.update(output)
    #     if output['patch_images'] != None:
    #         item_dict.update({'patch_images': self.patch_resize_transform(
    #             Image.open(BytesIO(base64.b64decode(self.table['image'][self.imageid_to_arrowid[output['patch_images']]].as_py())))
    #             )})  # 3,256,256  
    #     else:
    #         item_dict.update({'patch_masks': [False] * (self.resolution // 16) * (self.resolution // 16),
    #                           'patch_images': torch.ones(3,self.resolution, self.resolution) })
            
    #     return {key: torch.tensor(value) for key, value in item_dict.items()}