from clapt.datapipelining import (
    batch_collation,
    trainingsample_creation,
    wikipedia_loading,
    decoder_passing,
)
from clapt import modeling
import ray.data
import transformers


def create_trainingdatasets(
    MODEL_NAME: str = modeling.MODEL_NAME,
) -> ray.data.Dataset:
    return (
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
            batch_size=2,
            num_gpus=1,
            concurrency=4,  # 4x A100
        )
    )


def create_dataloaders(dataset: ray.data.Dataset, batch_size: int):
    return dataset.iter_torch_batches(
        batch_size=batch_size,
        collate_fn=batch_collation.pad_batch,
    )
