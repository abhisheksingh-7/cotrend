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
