from typing import Any, Optional, List, Tuple, Dict
import copy
import torch as T
import torch.nn as nn
import torch.distributed as dist
from torch.utils import data as torch_data
import lightning as L
from transformers import AutoModelForCausalLM, AutoConfig
from loguru import logger

from clapt import modeling
from clapt import datapipelining
from clapt.datapipelining import datamodels


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
        # self.base_llm = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.llm_config = AutoConfig.from_pretrained(base_model_id)
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.encoder_q = modeling.CLAPTHead(self.llm_config)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # for param in self.base_llm.parameters():
        #     param.requires_grad = False

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
        self, batch: Dict[str, T.Tensor], batch_idx: int
    ) -> T.Tensor | None:
        loss = self(**batch)
        return loss

    def forward(
        self,
        training_batch: datamodels.TrainingBatch,
        key_ids: Optional[T.Tensor] = None,
        query_ids: Optional[T.Tensor] = None,
    ) -> Tuple[T.Tensor, ...]:
        key_tensor = training_batch.key_tensor
        key_mask = training_batch.key_mask
        query_tensor = training_batch.query_tensor
        query_mask = training_batch.query_mask

        # with T.no_grad():
        #     if key_tensor is None:
        #         key_tensor = self.base_llm(
        #             input_ids=key_ids, attention_mask=key_mask
        #         ).hidden_states[-1]
        #     if query_tensor is None:
        #         query_tensor = self.base_llm(
        #             input_ids=query_ids, attention_mask=query_mask
        #         ).hidden_states[-1]

        bsz = query_tensor.size(0)
        q = self.encoder_q(key_tensor, key_mask)

        with T.no_grad():
            self._momentum_update_key_encoder()

            if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                self.encoder_k.eval()

            k = self.encoder_k(query_tensor, query_mask)
        logits = self._compute_logits(q, k) / self.temperature
        labels = T.zeros(bsz, dtype=T.long).cuda()
        loss = nn.functional.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing
        )
        self._dequeue_and_enqueue(k)
        print(loss)
        return loss

    def configure_optimizers(self):
        return T.optim.AdamW(
            self.parameters(), lr=5e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01
        )


# class CLAPTModel(L.LightningModule):
#     def __init__(self, name_or_path: str) -> None:
#         super().__init__()
#         self.name_or_path = name_or_path
#         self.model = modeling.CLAPT(name_or_path)

#     def configure_model(self) -> None:
#         if self.model is not None:
#             return
#         self.model = modeling.CLAPT(self.name_or_path)

#     def configure_optimizers(self):
#         return T.optim.AdamW(
#             self.parameters(), lr=5e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01
#         )

#     def forward(
#         self,
#         input_ids: T.LongTensor = None,
#         attention_mask: Optional[T.Tensor] = None,
#         position_ids: Optional[T.LongTensor] = None,
#         past_key_values: Optional[List[T.FloatTensor]] = None,
#         inputs_embeds: Optional[T.FloatTensor] = None,
#         labels: Optional[T.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[T.LongTensor] = None,
#         decoder_embeds: Optional[T.Tensor] = None,
#     ) -> Any:
#         return self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=True,
#             return_dict=True,
#             cache_position=cache_position,
#             decoder_embeds=decoder_embeds,
#         )


@T.no_grad()
def gather_nograd(x: T.Tensor) -> T.Tensor:
    if not dist.is_initialized():
        return x
    x_gather = [T.ones_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(x_gather, x, async_op=False)

    x_gather = T.cat(x_gather, dim=0)
    return x_gather


def main() -> None:
    train_data = datapipelining.create_trainingdatasets(modeling.MODEL_NAME)
    train_dataloader = datapipelining.create_dataloaders(train_data, batch_size=16)
    model = MoCo(
        modeling.MODEL_NAME,
        queue_size=6552,
        momentum=0.999,
        temperature=1.0,
        label_smoothing=0.0,
    )
    trainer = L.Trainer(num_nodes=1, max_epochs=10)
    trainer.fit(model=model, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    main()
