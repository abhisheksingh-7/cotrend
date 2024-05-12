from clapt.datapipelining import datamodels, padding, validation


@validation.validate_batch
def pad_batch(batches: list[datamodels.TrainingBatch]) -> datamodels.TrainingBatch:
    keys = padding.collate_batches(map(padding.MaskedInputs.from_keys, batches))
    queries = padding.collate_batches(map(padding.MaskedInputs.from_queries, batches))
    return datamodels.TrainingBatch(
        key_tensor=keys.input_ids,
        key_mask=keys.attention_mask,
        query_tensor=queries.input_ids,
        query_mask=queries.attention_mask,
    )


if __name__ == "__main__":
    import json

    from clapt.datapipelining import (
        trainingsample_creation,
        wikipedia_loading,
        decoder_passing,
    )
    from clapt import modeling
    import transformers

    MODEL_NAME = modeling.MODEL_NAME
    dataset = (
        wikipedia_loading.create_wikipedia_dataset()
        .map(
            trainingsample_creation.TrainingSampleFactory,  # type: ignore
            fn_constructor_args=(
                trainingsample_creation.AugmentationConfig(),
                transformers.AutoTokenizer.from_pretrained(MODEL_NAME),
            ),
            concurrency=8,
        )
        .map_batches(
            decoder_passing.LlamaLastHiddenStateExtractor,
            fn_constructor_args=(MODEL_NAME,),
            batch_size=1,
            concurrency=4,
            num_gpus=1,
        )
    )
    dataloader = dataset.iter_torch_batches(
        batch_size=32,
        collate_fn=pad_batch,
    )
    for batch in dataloader:
        print(json.dumps(batch, indent=2))
