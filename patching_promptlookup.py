from transformers.generation.utils import *
import torch
from torch import LongTensor, FloatTensor
from transformers.generation.candidate_generator import CandidateGenerator

from transformers.generation import GenerationConfig, LogitsProcessorList, CandidateGenerator, PromptLookupCandidateGenerator, AssistedCandidateGenerator
from transformers import PreTrainedModel, VisionEncoderDecoderModel
from transformers.models.kosmos2_5.modeling_kosmos2_5 import Kosmos2_5TextForCausalLM
from mugat.nougat.modeling_mbart import MBartForCausalLM
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.generation import candidate_generator

class PromptLookupCandidateGeneratiorMod(CandidateGenerator):
    def __init__(
        self,
        pdf_text_ids: LongTensor,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = None,
        max_length: int = 20,
    ) -> None:
        self.pdf_text_ids = pdf_text_ids
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size if max_matching_ngram_size else 2
        self.max_length = max_length

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

    def get_candidates(self, input_ids: LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        input_length = input_ids.size(1)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        if self.max_length == input_length + 1:
            return input_ids, None

        chosen_ids = None
        match_found = False
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 1, -1):
            # Create sliding windows of size ngram_size
            windows = self.pdf_text_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length, self.max_length)

                if start_idx < end_idx:
                    chosen_ids = self.pdf_text_ids[0, start_idx:end_idx]
                    match_found = True
                    break
            if match_found:
                break

        if chosen_ids is None or len(chosen_ids) == 0:
            # In case we didn't find a match return the input sequence unchanged, reverts back to autoregressive decoding
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids, None
    
    def update_candidate_strategy(self, input_ids: LongTensor, scores: FloatTensor, num_matches: int):
        # Currently does nothing
        #if num_matches > 0:
        #    print(num_matches)
        return
    

def _get_candidate_generator(
    self,
    generation_config: GenerationConfig,
    input_ids: torch.LongTensor,
    inputs_tensor: torch.Tensor,
    assistant_model: "PreTrainedModel",
    logits_processor: LogitsProcessorList,
    target_tokenizer: "PreTrainedTokenizerBase",
    assistant_tokenizer: "PreTrainedTokenizerBase",
    model_kwargs: Dict,
) -> CandidateGenerator:
    """
    Returns the candidate generator to be used in `assisted_generation`
    """
    different_tokenizers = all(v is not None for v in (assistant_model, target_tokenizer, assistant_tokenizer))
    if "pdf_text_ids" in model_kwargs:
        candidate_generator = PromptLookupCandidateGeneratiorMod(
            pdf_text_ids=model_kwargs["pdf_text_ids"].to(input_ids.device),
            num_output_tokens=generation_config.prompt_lookup_num_tokens,
            max_matching_ngram_size=generation_config.max_matching_ngram_size,
            max_length=generation_config.max_length,
        )
    elif generation_config.prompt_lookup_num_tokens is not None:
        candidate_generator = PromptLookupCandidateGenerator(
            num_output_tokens=generation_config.prompt_lookup_num_tokens,
            max_matching_ngram_size=generation_config.max_matching_ngram_size,
            max_length=generation_config.max_length,
        )
    elif different_tokenizers:
        candidate_generator = AssistedCandidateGeneratorDifferentTokenizers(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            target_tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
        )
    else:
        candidate_generator = AssistedCandidateGenerator(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
        )
    return candidate_generator


def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
    """Validates model kwargs for generation. Generate argument typos will also be caught here."""
    # If a `Cache` instance is passed, checks whether the model is compatible with it
    if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
        raise ValueError(
            f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
            "check the model documentation for supported cache formats."
        )

    # Excludes arguments that are handled before calling any model function
    if self.config.is_encoder_decoder:
        for key in ["decoder_input_ids"]:
            model_kwargs.pop(key, None)

    unused_model_args = []
    model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
    # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
    # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
    if "kwargs" in model_args or "model_kwargs" in model_args:
        model_args |= set(inspect.signature(self.forward).parameters)

    # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
    if self.config.is_encoder_decoder:
        base_model = getattr(self, self.base_model_prefix, None)

        # allow encoder kwargs
        encoder = getattr(self, "encoder", None)
        # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
        # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
        # TODO: A better way to handle this.
        if encoder is None and base_model is not None:
            encoder = getattr(base_model, "encoder", None)

        if encoder is not None:
            encoder_model_args = set(inspect.signature(encoder.forward).parameters)
            model_args |= encoder_model_args

        # allow decoder kwargs
        decoder = getattr(self, "decoder", None)
        if decoder is None and base_model is not None:
            decoder = getattr(base_model, "decoder", None)

        if decoder is not None:
            decoder_model_args = set(inspect.signature(decoder.forward).parameters)
            model_args |= {f"decoder_{x}" for x in decoder_model_args}

        # allow assistant_encoder_outputs to be passed if we're doing assisted generating
        if "assistant_encoder_outputs" in model_kwargs:
            model_args |= {"assistant_encoder_outputs"}

    if "pdf_text_ids" in model_kwargs:
        model_args |= {"pdf_text_ids"}

    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)

    if unused_model_args:
        raise ValueError(
            f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
            " generate arguments will also show up in this list)"
        )
    

def _crop_past_key_values(model, past_key_values, max_length):
    """Crops the past key values up to a certain maximum length."""
    new_past = []
    if model.config.is_encoder_decoder:
        for idx in range(len(past_key_values)):
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :max_length, :],
                    past_key_values[idx][1][:, :, :max_length, :],
                    past_key_values[idx][2],
                    past_key_values[idx][3],
                )
            )
        past_key_values = tuple(new_past)
    # gptbigcode is special and stores kv in shape (batch_size, seq_len, dim), if it's a multi_query model
    elif "gptbigcode" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "gptbigcode" in model.config.architectures[0].lower()
    ):
        if model.config.multi_query:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :max_length, :]
        else:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :, :max_length, :]
    elif model.__class__.__name__.lower().startswith("mllama") or (
        model.config.architectures is not None and model.config.architectures[0].lower().startswith("mllama")
    ):
        # DynamicCache but do not filter cross attention layers.
        if max_length < 0:
            max_length = past_key_values.get_seq_length() - abs(max_length)

        if past_key_values.get_seq_length() <= max_length:
            return past_key_values
        
        past_key_values._seen_tokens = max_length

        for idx in range(len(past_key_values.key_cache)):
            if past_key_values.key_cache[idx] != [] and "Cross" not in model.language_model.model.layers[idx].__class__.__name__:
                past_key_values.key_cache[idx] = past_key_values.key_cache[idx][..., :max_length, :]
                past_key_values.value_cache[idx] = past_key_values.value_cache[idx][..., :max_length, :]

    elif isinstance(past_key_values, DynamicCache):
        past_key_values.crop(max_length)
    elif past_key_values is not None:
        for idx in range(len(past_key_values)):
            if past_key_values[idx] != ([], []):
                new_past.append(
                    (
                        past_key_values[idx][0][:, :, :max_length, :],
                        past_key_values[idx][1][:, :, :max_length, :],
                    )
                )
            else:
                new_past.append((past_key_values[idx][0], past_key_values[idx][1]))
        past_key_values = tuple(new_past)
    return past_key_values


def apply_patching_promptlookup():
    VisionEncoderDecoderModel._get_candidate_generator = _get_candidate_generator
    VisionEncoderDecoderModel._validate_model_kwargs = _validate_model_kwargs
    Kosmos2_5TextForCausalLM._get_candidate_generator = _get_candidate_generator
    Kosmos2_5TextForCausalLM._validate_model_kwargs = _validate_model_kwargs
    MBartForCausalLM._get_candidate_generator = _get_candidate_generator
    MBartForCausalLM._validate_model_kwargs = _validate_model_kwargs
    MllamaForConditionalGeneration._get_candidate_generator = _get_candidate_generator
    MllamaForConditionalGeneration._validate_model_kwargs = _validate_model_kwargs
    Qwen2VLForConditionalGeneration._get_candidate_generator = _get_candidate_generator
    Qwen2VLForConditionalGeneration._validate_model_kwargs = _validate_model_kwargs
    candidate_generator._crop_past_key_values.__code__ = _crop_past_key_values.__code__