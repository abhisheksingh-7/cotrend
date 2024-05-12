from typing import Optional, List, Union, Tuple
import torch as T
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_ENCODER_LAYERS = 2


class CLAPTHead(nn.Module):
    def __init__(
        self, llm_config: AutoConfig, num_layers: int = NUM_ENCODER_LAYERS
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_config.hidden_size,
            nhead=llm_config.num_attention_heads,
            dim_feedforward=llm_config.hidden_size,
            batch_first=True,
        )
        self.query_vec = nn.Embedding(1, embedding_dim=llm_config.hidden_size)
        if num_layers < 1:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, decoder_embeds: T.Tensor, attention_mask: T.Tensor) -> T.Tensor:
        encodings, mask = self._get_encoding_with_query_vec(
            decoder_embeds, attention_mask
        )
        return self.encoder(src=encodings, src_key_padding_mask=mask)[:, 0, :]

    def _get_encoding_with_query_vec(
        self, decoder_embeds: T.Tensor, attention_mask: Optional[T.Tensor]
    ) -> Tuple[T.Tensor, Optional[T.Tensor]]:
        encodings = T.cat(
            (
                self.query_vec.weight.unsqueeze(0).expand(len(decoder_embeds), -1, -1),
                decoder_embeds,
            ),
            dim=1,
        )
        if attention_mask is not None:
            query_mask = T.ones(
                (attention_mask.shape[0], 1), dtype=T.bool, device=attention_mask.device
            )
            mask = ~T.cat([query_mask, attention_mask], dim=1).type(T.bool)
        else:
            mask = attention_mask
        return encodings, mask


class CLAPT(nn.Module):
    def __init__(
        self,
        decoder_with_lm: AutoModelForCausalLM,
        num_layers: int = NUM_ENCODER_LAYERS,
    ) -> None:
        super().__init__()
        self.decoder_with_lm = decoder_with_lm
        self.clapt_head = CLAPTHead(self.decoder_with_lm.config, num_layers)

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if decoder_embeds is None:
            decoder_embeds = self.get_decoder_last_hidden_state(
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
        encodings, mask = self.get_encoding_with_query_vec(
            decoder_embeds, attention_mask
        )
        return self.encoder(src=encodings, src_key_padding_mask=mask)[:, 0, :]

    def get_decoder_last_hidden_state(
        self,
        input_ids: T.Tensor = None,
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
    ) -> T.Tensor:
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
        return decoder_embeds

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
    llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = CLAPT(llm).to(device)
    print(model)
    inputs = tokenizer(
        ["Hello there sir, are you CLAPT?"], return_tensors="pt", padding="longest"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(model(**inputs).shape)


if __name__ == "__main__":
    main()
