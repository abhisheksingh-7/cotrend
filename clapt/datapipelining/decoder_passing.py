import torch

from clapt import modeling
from clapt.datapipelining import datamodels, padding, validation


class LlamaLastHiddenStateExtractor:
    def __init__(
        self,
        model_name: str = modeling.MODEL_NAME,
        decoder_device: torch.device = torch.device("cuda"),
        output_device: torch.device = torch.device("cpu"),
    ) -> None:
        self.decoder_device = decoder_device
        self.output_device = output_device
        self.model = modeling.CLAPT(model_name, num_layers=0).to(decoder_device)

    @validation.validate_batch
    def __call__(
        self, dp_batch: list[datamodels.TrainingDataPoint]
    ) -> datamodels.TrainingBatch:
        batch = collate_batch(dp_batch)
        key_embeds = self._get_decoder_embeds(
            batch.key_tensor.to(self.decoder_device),
            batch.key_mask.to(self.decoder_device),
        )
        query_embeds = self._get_decoder_embeds(
            batch.query_tensor.to(self.decoder_device),
            batch.query_mask.to(self.decoder_device),
        )
        return datamodels.TrainingBatch(
            key_tensor=key_embeds,
            key_mask=batch.key_mask,
            query_tensor=query_embeds,
            query_mask=batch.query_mask,
        ).to(self.output_device)

    def _get_decoder_embeds(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.model.get_decoder_last_hidden_state(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )


def collate_batch(
    batch: list[datamodels.TrainingDataPoint],
) -> datamodels.TrainingBatch:
    keys = padding.collate_samples(dp.key for dp in batch)
    queries = padding.collate_samples(dp.query for dp in batch)
    return datamodels.TrainingBatch(
        key_tensor=keys.input_ids,
        key_mask=keys.attention_mask,
        query_tensor=queries.input_ids,
        query_mask=queries.attention_mask,
    )


if __name__ == "__main__":
    import json

    from clapt.datapipelining import trainingsample_creation, wikipedia_loading

    MODEL_NAME = modeling.MODEL_NAME
    dataset = (
        wikipedia_loading.create_wikipedia_dataset()
        .map(
            trainingsample_creation.TrainingSampleFactory.from_model_name(MODEL_NAME),
            concurrency=8,
        )
        .map_batches(
            LlamaLastHiddenStateExtractor(MODEL_NAME),
            batch_size=4,
            concurrency=2,
            num_gpus=1,
        )
    )

    datapoint = dataset.take(1)
    print(json.dumps(datapoint, indent=2))
