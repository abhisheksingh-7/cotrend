from typing import Optional, List, Union, Tuple
import torch as T
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_ENCODER_LAYERS = 2


class CLAPT(nn.Module):
    def __init__(self, name_or_path: str, num_layers: int = NUM_ENCODER_LAYERS) -> None:
        super().__init__()
        self.name_or_path = name_or_path
        self.decoder_with_lm = AutoModelForCausalLM.from_pretrained(name_or_path)
        self.decoder_with_lm.resize_token_embeddings(
            self.decoder_with_lm.config.vocab_size + 1
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_with_lm.config.hidden_size,
            nhead=self.decoder_with_lm.config.num_attention_heads,
            dim_feedforward=self.decoder_with_lm.config.hidden_size,
            batch_first=True,
        )
        self.query_vec = nn.Embedding(
            1, embedding_dim=self.decoder_with_lm.config.hidden_size
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        lm_outputs: CausalLMOutputWithPast = self.decoder_with_lm(
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
        )
        decoder_embeds = lm_outputs.hidden_states[-1]
        encodings, mask = self.get_encoding_with_query_vec(
            decoder_embeds, attention_mask
        )
        embedding = self.encoder(src=encodings, src_key_padding_mask=mask)[:, 0, :]
        return embedding

    def get_encoding_with_query_vec(
        self, decoder_embeds: T.Tensor, attention_mask: T.Tensor
    ) -> Tuple[T.Tensor, Optional[T.Tensor]]:
        encodings = T.cat(
            (
                self.query_vec.weight.unsqueeze(0).expand(len(decoder_embeds), -1, -1),
                decoder_embeds,
            ),
            dim=1,
        )
        query_mask = T.ones(
            (attention_mask.shape[0], 1), dtype=T.bool, device=attention_mask.device
        )
        mask = ~T.cat([query_mask, attention_mask], dim=1).type(T.bool)
        return encodings, mask


def main() -> None:
    device = T.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    model = CLAPT(MODEL_NAME).to(device)
    print(model)
    inputs = tokenizer(
        ["Hello there sir, are you CLAPT?"], return_tensors="pt", padding="longest"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(model(**inputs).shape)


if __name__ == "__main__":
    main()
