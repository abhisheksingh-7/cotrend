from typing import Any, Optional, List
import torch as T
import lightning as L
from transformers import AutoModelForCausalLM

from clapt import modeling


class CLAPTData(L.LightningDataModule):
    def __init__(self) -> None: ...
    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def train_dataloader(self) -> Any:
        return super().train_dataloader()

    def val_dataloader(self) -> Any:
        return super().val_dataloader()

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)


class MoCo(L.LightningModule):
    def __init__(
        self, base_model_id: str, queue_size: int, momentum: float, temperature: float
    ) -> None:
        super().__init__()
        self.base_llm = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.encoder_q = None
        self.encoder_k = None

    def configure_model(self) -> None:
        if self.model is not None:
            return
        self.model = modeling.CLAPT(self.name_or_path)


class CLAPTModel(L.LightningModule):
    def __init__(self, name_or_path: str) -> None:
        super().__init__()
        self.name_or_path = name_or_path
        self.model = modeling.CLAPT(name_or_path)

    def configure_model(self) -> None:
        if self.model is not None:
            return
        self.model = modeling.CLAPT(self.name_or_path)

    def configure_optimizers(self):
        return T.optim.AdamW(
            self.parameters(), lr=5e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01
        )

    def forward(
        self,
        input_ids: T.LongTensor = None,
        attention_mask: Optional[T.Tensor] = None,
        position_ids: Optional[T.LongTensor] = None,
        past_key_values: Optional[List[T.FloatTensor]] = None,
        inputs_embeds: Optional[T.FloatTensor] = None,
        labels: Optional[T.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[T.LongTensor] = None,
        decoder_embeds: Optional[T.Tensor] = None,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
            decoder_embeds=decoder_embeds,
        )


def main() -> None:
    model = CLAPTModel(modeling.MODEL_NAME)
    datamodule = None
    trainer = L.Trainer(
        num_nodes=1,
        max_epochs=10,
    )
    trainer.fit(model=model, datamodule=datamodule)
