from typing import Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
import torch.nn as nn
import torch

import pytorch_lightning as pl
import pandas as pd

from util.others.my_metrics import Accuracy
# from torchmetrics import F1Score
from util.others.dist_utils import is_main_process

import csv

from transformers import (
    get_cosine_schedule_with_warmup
)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels, dropout):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
        self.gelu = torch.nn.GELU()
        
        self._init_weights(self.dense)
        self._init_weights(self.out_proj)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = torch.sum(features, dim=1) / features.shape[1]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        # x = self.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SourceRetrievalModule(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.save_hyperparameters()
        self._config = _config
        self.training_loss = None
        self.val_loss = []
        self.pred = []
        self.outlier = {}
                
        self.model_config = AutoConfig.from_pretrained(_config['model_name'])
        self.model = AutoModel.from_pretrained(_config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(_config['model_name'])
            
        # self.metric = F1Score('multilabel', num_labels=_config['num_labels'])
        self.metric = Accuracy()
        self.noise_sampler = torch.distributions.normal.Normal(loc=0.0, scale=1e-5)
        self.r3f_lambda = 1.0
        
        self.classification_head = ClassificationHead(config=self.model_config, num_labels=_config['num_labels'], dropout=0.1) #s # 59457 # 이걸로 head_layer
        # self.classification_layer = nn.Linear(self.model_config.d_model, _config['num_labels'])
        
        self.loss_fct = nn.CrossEntropyLoss()
        
        self.ids = []
        self.results = []
        
    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
        ) / noised_logits.size(0)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # # logits = self.classification_layer(outputs.last_hidden_state[:,0,:])
        
        # # ver2
        logits = self.classification_head.forward(outputs.last_hidden_state)
        
        if labels == None:
            loss = None
        else:
            loss = self.loss_fct(logits, labels)
        
        
        # r3f
        # token_embeddings = self.model.embeddings(input_ids=input_ids)
        # output = self.model(input_ids=None, attention_mask=attention_mask, inputs_embeds=token_embeddings)
        # noise = self.noise_sampler.sample(sample_shape=token_embeddings.shape).to(token_embeddings)
        # noised_embeddings = token_embeddings.detach().clone() + noise
        # noised_output = self.model(
        #         input_ids=None,
        #         inputs_embeds=noised_embeddings,
        #         attention_mask=attention_mask,
        #     )
        
        # logits = self.classification_head.forward(output.last_hidden_state)
        # noised_logits = self.classification_head.forward(noised_output.last_hidden_state)
        # symm_kl = self._get_symm_kl(noised_logits, logits)
        
        # if labels == None:
        #     loss = None
        # else:
        #     loss = self.loss_fct(logits, labels) + (self.r3f_lambda * symm_kl)
        
        
        # -- for outlier extraction ---------------------
        # preds = logits.argmax(dim=-1)
        # preds = preds[labels != -100]
        # labels = labels[labels != -100]
        # if labels.numel() == 0:
        #     return 1

        # assert preds.shape == labels.shape
        
        # title = self.tokenizer.batch_decode(input_ids)
        # idx = list(filter(lambda x : torch.eq(preds, labels).tolist()[x] == False, range(len(preds))))
   
        # for i in idx:
        #     if title[i] not in self.outlier:
        #         self.outlier[title[i]] = 1
        #     else:
        #         self.outlier[title[i]] += 1
        # -----------------------------------------------
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
        
        
    def training_step(self, batch, batch_idx):
        output = self(**batch)
        self.training_loss = output.loss
        
        self.log(f"train/loss", output.loss)
        
        return output.loss
    
    def training_epoch_end(self, outputs) -> None:
        # -- for outlier extraction ----------
        with open("./outlier.csv", 'w', encoding='UTF-8') as f:
            w = csv.writer(f)
            w.writerow(self.outlier.keys())
            w.writerow(self.outlier.values())
        # ------------------------------------
    
    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.metric.update(output.logits, batch['labels'])
        self.val_loss.append(output.loss.tolist())
        
        self.log(f"val/loss", output.loss)

    def on_validation_epoch_end(self):
        acc = self.metric.compute().tolist()
        self.metric.reset()
        
        self.log(f"val/acc", acc)
        print(f"val_loss : {sum(self.val_loss) / len(self.val_loss)}")
        print(f"Accuracy : {acc}")
      
        self.val_loss.clear()
        
        # -- for outlier extraction ----------
        with open("./outlier.csv", 'w', encoding='UTF-8') as f:
            w = csv.writer(f)
            w.writerow(self.outlier.keys())
            w.writerow(self.outlier.values())
        # ------------------------------------
      
    def test_step(self, batch, batch_id) -> None:
        output = self(**batch)
        self.pred += output.logits.argmax(dim=-1).tolist()
        
    def test_epoch_end(self, outputs) -> None:
        df = pd.DataFrame({"pred":self.pred})
        df.to_csv('/home2/yangcw/book/pred.csv', index=False)
        
    # def test_step(self, batch, batch_idx):
    #     output = self(**batch)
    #     logits = output.logits
    #     preds = logits.argmax(dim=-1)
        
    #     self.results += preds.tolist()
    #     self.ids += batch['id'].tolist()
        
    #     torch.distributed.barrier()
        
    # def test_epoch_end(self, outs):
    #     if is_main_process:   # 1 gpu only
    #         import json
    #         with open(self._config['test_dataset_path']) as f:
    #             dataset = json.load(f)
    #             id_to_qid = {}

    #             for qid in dataset.keys():
    #                 id_to_qid[dataset[qid]['id']] = qid
            
    #         for id, result in zip(self.ids, self.results):
    #             dataset[id_to_qid[id]]['modality'] = result
                
    #         with open(self._config['test_dataset_path']+'result', 'w') as outfile:
    #             json.dump(dataset, outfile, indent=4)           
            
    #         # print(self.results)
    #     torch.distributed.barrier()        
        
        
    def configure_optimizers(self):
        param_optimizer = self.named_parameters()
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams._config['lr'], betas=(0.9, 0.999))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams._config['warmup_steps'], num_training_steps=self.hparams._config['max_steps']
        )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )
