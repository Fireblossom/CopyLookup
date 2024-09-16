import torch
from transformers.models.kosmos2_5.modeling_kosmos2_5 import  Kosmos2_5TextForCausalLM, create_position_ids_from_input_ids


def prepare_inputs_for_generation(
    self,
    input_ids,
    image_embeds=None,
    image_embeds_position_mask=None,
    past_key_values=None,
    attention_mask=None,
    use_cache=None,
    **model_kwargs,
):
    input_shape = input_ids.shape
    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_shape)

    position_ids = None

    # cut input_ids if past_key_values is used
    if past_key_values is not None:
        past_length = past_key_values[0][0].size(2)
        new_length = input_shape[1]
        add_length = new_length - past_length 

        position_ids = create_position_ids_from_input_ids(
            input_ids,
            padding_idx=self.config.pad_token_id,
            past_key_values_length=0,
        )[:, -add_length:]

        input_ids = input_ids[:, -add_length:]
        # the image info. is already encoded into the past keys/values
        image_embeds = None
        image_embeds_position_mask = None
    elif image_embeds_position_mask is not None:
        # appending `False` to `image_embeds_position_mask` (because `input_ids` grows during generation)
        batch_size, seq_len = input_ids.size()
        mask_len = image_embeds_position_mask.size()[-1]
        image_embeds_position_mask = torch.cat(
            (
                image_embeds_position_mask,
                torch.zeros(size=(batch_size, seq_len - mask_len), dtype=torch.bool, device=input_ids.device),
            ),
            dim=1,
        )

    return {
        "input_ids": input_ids,
        "image_embeds": image_embeds,
        "image_embeds_position_mask": image_embeds_position_mask,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": use_cache,
    }


def apply_patching_kosmos2_5():
    Kosmos2_5TextForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation