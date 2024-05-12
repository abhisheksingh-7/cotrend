import copy
from lightning.fabric.utilities import rank_zero
from typing import Tuple
import numpy as np
import os
import datetime
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint
import functools
import lightning as L
import pydantic
import torch as T
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from lightning.pytorch import loggers
from torch.utils import data as torch_data
from transformers import AutoConfig, AutoModelForCausalLM
import ray.train.lightning
import ray.train.torch
from pytorch_metric_learning import miners, losses
from torch.utils.data.sampler import SubsetRandomSampler


from clapt import datapipelining, modeling
from clapt.datapipelining import datamodels, torch_dataloading

TRAIN_SHARD = "train"


class ClaptSap(L.LightningModule):
    def __init__(self, base_model_id: str) -> None:
        super().__init__()
        self.base_model_id = base_model_id
        self.base_llm = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.base_llm.eval()
        self.llm_config = AutoConfig.from_pretrained(base_model_id)
        self.encoder = modeling.CLAPTHead(self.llm_config)
        for param in self.base_llm.parameters():
            param.requires_grad = False
        self.miner = miners.TripletMarginMiner(0.2, "all")
        self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5)

    def validation_step(
        self,
        batch: Tuple[T.Tensor, ...],
        batch_idx: int,
    ) -> T.Tensor | None:
        query_toks1, query_toks2, labels = batch
        loss = self(query_toks1, query_toks2, labels)
        self.log(
            "eval/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(
        self,
        batch: Tuple[T.Tensor, ...],
        batch_idx: int,
    ) -> T.Tensor | None:
        query_toks1, query_toks2, labels = batch
        loss = self(query_toks1, query_toks2, labels)
        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def forward(
        self, query_toks1: T.Tensor, query_toks2: T.Tensor, labels: T.Tensor
    ) -> T.Tensor:
        term1_hidden = self.base_llm(
            **query_toks1, output_hidden_states=True
        ).hidden_states[-1]
        term2_hidden = self.base_llm(
            **query_toks2, output_hidden_states=True
        ).hidden_states[-1]
        query_embed1 = nn.functional.normalize(
            self.encoder(term1_hidden, query_toks1["attention_mask"]), dim=-1
        )
        query_embed2 = nn.functional.normalize(
            self.encoder(term2_hidden, query_toks2["attention_mask"]), dim=-1
        )
        query_embed = T.cat([query_embed1, query_embed2], dim=0)
        labels = T.cat([labels, labels], dim=0)
        hard_pairs = self.miner(query_embed, labels)
        return self.loss(query_embed, labels, hard_pairs)

    def configure_optimizers(self):
        return T.optim.AdamW(
            self.parameters(), lr=5e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01
        )


class MoCo(L.LightningModule):
    def __init__(
        self,
        base_model_id: str,
        queue_size: int,
        momentum: float,
        temperature: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.base_model_id = base_model_id
        self.base_llm = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.llm_config = AutoConfig.from_pretrained(base_model_id)
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.encoder_q = modeling.CLAPTHead(self.llm_config)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        for param in self.base_llm.parameters():
            param.requires_grad = False

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer(
            "queue", T.randn(self.llm_config.hidden_size, self.queue_size)
        )
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", T.zeros(1, dtype=T.long))

    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    def configure_model(self) -> None:
        if self.base_llm is not None:
            return
        self.base_llm = AutoModelForCausalLM.from_pretrained(self.base_model_id)

    @T.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = gather_nograd(keys.contiguous())
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert (
            self.queue_size % batch_size == 0
        ), f"{batch_size}, {self.queue_size}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    def _compute_logits(self, q, k):
        l_pos = T.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = T.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = T.cat([l_pos, l_neg], dim=1)
        return logits

    def training_step(
        self, batch: datamodels.TrainingBatch, batch_idx: int
    ) -> T.Tensor | None:
        batch = pydantic.TypeAdapter(datamodels.TrainingBatch).validate_python(batch)
        loss = self(batch)
        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def forward(
        self,
        batch: datamodels.TrainingBatch,
    ) -> Tuple[T.Tensor, ...]:
        self.base_llm.eval()
        batch = pydantic.TypeAdapter(datamodels.TrainingBatch).validate_python(batch)
        key_tensor = batch.key_tensor
        key_mask = batch.key_mask
        query_tensor = batch.query_tensor
        query_mask = batch.query_mask
        with T.no_grad():
            key_tensor = self.base_llm(
                input_ids=key_tensor, attention_mask=key_mask, output_hidden_states=True
            ).hidden_states[-1]
            query_tensor = self.base_llm(
                input_ids=query_tensor,
                attention_mask=query_mask,
                output_hidden_states=True,
            ).hidden_states[-1]
        bsz = query_tensor.size(0)
        q = self.encoder_q(query_tensor, query_mask)

        with T.no_grad():
            self._momentum_update_key_encoder()

            if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                self.encoder_k.eval()

            k = self.encoder_k(key_tensor, key_mask)
        q = T.nn.functional.normalize(q, dim=-1)
        k = T.nn.functional.normalize(k, dim=-1)
        logits = self._compute_logits(q, k) / self.temperature
        labels = T.zeros(bsz, dtype=T.long).cuda()
        loss = nn.functional.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing
        )
        self._dequeue_and_enqueue(k)
        return loss

    def configure_optimizers(self):
        return T.optim.AdamW(
            self.parameters(), lr=5e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01
        )


class SaveHead(L.Callback):
    def __init__(self, inference_checkpoint_path: str) -> None:
        super().__init__()
        self.inference_checkpoint_path = inference_checkpoint_path
        if rank_zero._get_rank() == 0:
            os.makedirs(self.inference_checkpoint_path, exist_ok=True)

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if trainer.is_global_zero:
            encoder = pl_module.encoder
            T.save(
                encoder.state_dict(),
                f"{self.inference_checkpoint_path}/checkpoint_epoch_{trainer.current_epoch}_step_{trainer.global_step}.pth",
            )


@T.no_grad()
def gather_nograd(x: T.Tensor) -> T.Tensor:
    if not dist.is_initialized():
        return x
    x_gather = [T.ones_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(x_gather, x, async_op=False)

    x_gather = T.cat(x_gather, dim=0)
    return x_gather


def train() -> None:
    DATA_PATH = (
        "/data/sap_synonyms.txt"
        if os.uname().nodename == "koozie-00"
        else "/nfs/scratch/data/sap_synonyms.txt"
    )
    dataset = torch_dataloading.SapBertDataset(DATA_PATH)
    validation_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(211)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    batch_size = 32
    num_nodes = 4
    tokenizer = AutoTokenizer.from_pretrained(modeling.MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    collate_fn = functools.partial(
        torch_dataloading.collate_fn_batch_encoding, tokenizer=tokenizer
    )
    train_loader = T.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn
    )
    validation_loader = T.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=collate_fn
    )
    model = ClaptSap(modeling.MODEL_NAME)
    run_name = f"clapt-{datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}"

    wandb_logger = loggers.WandbLogger(
        project="llama-hackathon",
        name=run_name,
    )
    model_checkpoint = ModelCheckpoint(dirpath="/nfs/scratch/data/clapt/full")
    trainer = L.Trainer(
        max_epochs=10,
        strategy="ddp",
        num_nodes=num_nodes,
        accelerator="gpu",
        logger=wandb_logger,
        default_root_dir="/nfs/scratch/data/clapt",
        log_every_n_steps=25,
        callbacks=[SaveHead("/nfs/scratch/data/clapt/head"), model_checkpoint],
    )
    wandb_logger.watch(model=model)

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )


if __name__ == "__main__":
    train()
