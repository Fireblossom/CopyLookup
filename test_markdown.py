"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['HF_HOME'] = '/pfss/mlde/users/cd58hofa/huggingface/'

import argparse
import json
import logging
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import time

from transformers import NougatProcessor, VisionEncoderDecoderModel
from transformers import AutoTokenizer, LayoutLMv3ImageProcessor, AutoProcessor, MllamaForConditionalGeneration, Kosmos2_5ForConditionalGeneration, Qwen2VLForConditionalGeneration
from mugat.nougat import NougatModel as MugatModel
from mugat.nougat.utils.checkpoint import get_checkpoint

from nougat.metrics import compute_metrics, split_text
from nougat.utils.device import move_to_device
from datasets import ClassLabel

from token_path_prediction.dataset.code_doc_dataset import RainbowBankDataset
from token_path_prediction.dataset.feature_processor.ernie_processor import ErnieProcessor as ErniePLProcessor
from ernie_layout_pytorch.networks import ErnieLayoutConfig, set_config_for_extrapolation
from ernie_layout_pytorch.networks import exErnieLayoutForTokenClassification, ErnieLayoutTokenizerFast, ErnieLayoutProcessor


def test(args):
    if args.lookup_decoding == 'copylookup':
        from patching_copylookup import apply_patching_copylookup, build_copy_seq
        apply_patching_copylookup()
    elif args.lookup_decoding == 'promptlookup':
        from patching_promptlookup import apply_patching_promptlookup
        apply_patching_promptlookup()

    labels = ClassLabel(names=['DELETE', 'KEEP'])
    filter_config = ErnieLayoutConfig.from_pretrained('Norm/ERNIE-Layout-Pytorch', output_hidden_states=True)
    filter_config.pretrained_model_path = 'Norm/ERNIE-Layout-Pytorch'
    filter_config.adapter_pth_name = 'ernie_ckpt_caption'
    filter_config.num_labels = labels.num_classes
    filter_config.label2id = labels._str2int
    filter_config.id2label = labels._int2str
    filter_config.use_flash_attn = True  # use flash attention
    set_config_for_extrapolation(filter_config)

    filter_model = exErnieLayoutForTokenClassification.from_pretrained(
        config=filter_config,
        pretrained_model_name_or_path=filter_config.pretrained_model_path,
        ignore_mismatched_sizes=True
    )
    device = "cuda:0"
    dtype = torch.float32
    filter_model.load_adapter(filter_config.adapter_pth_name)
    filter_model = move_to_device(filter_model, bf16=False, cuda=device)
    filter_model.eval()

    tokenizer_config = torch.load('tokenizer_config.pt')
    tokenizer_config["mask_token"] = "<mask>"
    tokenizer_config["unk_token"] = "<unk>"
    tokenizer_config["pad_token"] = "<pad>"
    tokenizer_config["cls_token"] = "<s>"
    tokenizer_config["sep_token"] = "</s>"
    tokenizer_config["tokenizer_file"] = "tokenizer.json"
    tokenizer = ErnieLayoutTokenizerFast(**tokenizer_config)

    tokenizer.padding_side = 'right'
    tokenizer.only_label_first_subword = False

    ernie_image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    ernie_processor = ErnieLayoutProcessor(image_processor=ernie_image_processor, tokenizer=tokenizer)
    data_processor = ErniePLProcessor(
        data_dir='/raid/duan/cd58hofa/rainbow_bank/', 
        ernie_processor=ernie_processor, 
        max_length=1024
    )

    if args.vl_model == 'nougat':
        repo = "facebook/nougat-base"
        processor = NougatProcessor.from_pretrained(repo)
        model = VisionEncoderDecoderModel.from_pretrained(repo, device_map=device, torch_dtype=dtype)
    elif args.vl_model == 'mugat':
        repo = "/pfss/mlde/users/cd58hofa/mugat_ckpt" # hard coded local ckpt
        processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        model = MugatModel.from_pretrained(get_checkpoint(repo), device_map=device, torch_dtype=dtype)
        model.config.max_length=1024
    elif args.vl_model == 'kosmos_2.5':
        from patching_kosmos2_5 import apply_patching_kosmos2_5
        apply_patching_kosmos2_5()
        repo = "microsoft/kosmos-2.5"
        processor = AutoProcessor.from_pretrained(repo)
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, device_map=device, torch_dtype=dtype)
    elif args.vl_model == 'llama_3.2':
        repo = "/raid/duan/cache/" # hard coded local ckpt
        processor = AutoProcessor.from_pretrained(repo)
        model = MllamaForConditionalGeneration.from_pretrained(repo, device_map=device, torch_dtype=dtype)

    elif args.vl_model == 'qwen2-vl':
        repo = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
        processor = AutoProcessor.from_pretrained(repo)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        raise NotImplementedError

    dataset = RainbowBankDataset(
        data_dir=args.data_dir,
        data_processor=data_processor,
        dataset_name=args.dataset_name # 'quant_ph.txt'
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=None,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
        num_workers=6
    )

    img_path = Path(args.data_dir)/'images'

    for idx, sample in tqdm(enumerate(data_loader), total=len(data_loader)):
        if sample is None:
            continue
        sample.pop("labels")
        texts = sample.pop('texts')
        uids = sample.pop('uid')

        # prepare inputs
        images = [Image.open(img_path/img) for img in uids]
        if args.vl_model == 'nougat':
            inputs = processor(
                images,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
        elif args.vl_model == 'mugat': # batch_size == 1
            image_tensor = model.encoder.prepare_input(
                images[0], random_padding=False
            )
            uid = uids[0]
            page_num = int(uid.split('_')[-1].split('.')[0])
            next_page = "_".join(uid.split('_')[:-1]) + "_" + str(page_num+1) + ".png"
            prev_page = "_".join(uid.split('_')[:-1]) + "_" + str(page_num-1) + ".png"
            if (img_path/next_page).is_file():
                next_image_tensor = model.encoder.prepare_input(
                    Image.open(img_path/next_page), random_padding=False
                )
            else:
                next_image_tensor = torch.zeros(1)

            if (img_path/prev_page).is_file():
                prev_image_tensor = model.encoder.prepare_input(
                    Image.open(img_path/prev_page), random_padding=False
                )
            else:
                prev_image_tensor = torch.zeros(1)
        elif args.vl_model == 'kosmos_2.5':
            prompt = ["<md>"] * sample["input_ids"].size(0)
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            height, width = inputs.pop("height"), inputs.pop("width")
            inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
            inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
        elif args.vl_model == 'llama_3.2':
            prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>Convert the following PDF page to Markdown.\nReturn only the Markdown with no explanation text.\nLeave out any page numbers and redundant headers or footers.\nDo not include any code blocks (e.g. "```markdown" or "```") in the response.\nIf unable to parse, return an empty string.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            inputs = processor(text=[prompt], images=images, return_tensors="pt").to(model.device)
        elif args.vl_model == "qwen2-vl":
            prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Convert the following PDF page to Markdown.\nReturn only the Markdown with no explanation text.\nLeave out any page numbers and redundant headers or footers.\nDo not include any code blocks (e.g. "```markdown" or "```") in the response.\nIf unable to parse, return an empty string.<|im_end|>\n<|im_start|>assistant\n'
            inputs = processor(
                text=[prompt],
                images=images,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
        sample = {k: v.to(device) if v is not None else None for k, v in sample.items()}
        sample['pixel_values'] = sample['pixel_values'].half()

        # start decoding
        if not args.lookup_decoding:
            if args.vl_model == 'mugat':
                generated = model.inference(
                    image_tensors=image_tensor.to(device).unsqueeze(0),
                    prev_image_tensors=prev_image_tensor.to(device).unsqueeze(0),
                    next_image_tensors=next_image_tensor.to(device).unsqueeze(0),
                    early_stopping=False,
                )
            elif processor.tokenizer.unk_token_id is None:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                )
            if args.debug:
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(output_text)

        elif args.lookup_decoding == 'promptlookup':
            pdf_texts = [text[0] for text in texts]
            pdf_texts_ids = processor.tokenizer(' '.join(pdf_texts), return_tensors="pt").input_ids
            if args.vl_model == 'mugat':
                generated = model.inference(
                    image_tensors=image_tensor.to(device).unsqueeze(0),
                    prev_image_tensors=prev_image_tensor.to(device).unsqueeze(0),
                    next_image_tensors=next_image_tensor.to(device).unsqueeze(0),
                    early_stopping=False,
                    prompt_lookup_num_tokens=10,
                    max_matching_ngram_size=3,
                    pdf_text_ids = pdf_texts_ids,
                )
            elif processor.tokenizer.unk_token_id is None:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    prompt_lookup_num_tokens=10,
                    max_matching_ngram_size=3,
                    pdf_text_ids = pdf_texts_ids,
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    prompt_lookup_num_tokens=10,
                    max_matching_ngram_size=3,
                    pdf_text_ids = pdf_texts_ids,
                )
            if args.debug:
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(output_text)

        elif args.lookup_decoding == 'copylookup':
            
            filter_prediction = filter_model(**sample).logits
            filtered_texts = build_copy_seq(filter_prediction, sample, texts)

            copy_seqs = [processor.tokenizer(filtered_text).input_ids if filtered_text else [] for filtered_text in filtered_texts]
            if args.vl_model in ['nougat', 'mugat']:
                copy_seqs = [torch.LongTensor(seq[1:-1]) if seq else torch.LongTensor([]) for seq in copy_seqs[0]]
            else:
                copy_seqs = [torch.LongTensor(seq) if seq else torch.LongTensor([]) for seq in copy_seqs[0]]
            copy_seqs = [tensor for tensor in copy_seqs if tensor.size(0)>4]
            
            if args.vl_model == 'mugat':
                generated = model.inference(
                    image_tensors=image_tensor.to(device).unsqueeze(0),
                    prev_image_tensors=prev_image_tensor.to(device).unsqueeze(0),
                    next_image_tensors=next_image_tensor.to(device).unsqueeze(0),
                    early_stopping=False,
                    copy_seqs = copy_seqs,
                    max_matching_ngram_size=3,
                    prompt_lookup_num_tokens=10,
                )
            elif processor.tokenizer.unk_token_id is None:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    copy_seqs = copy_seqs,
                    max_matching_ngram_size=3,
                    prompt_lookup_num_tokens=10,
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    copy_seqs = copy_seqs,
                    max_matching_ngram_size=3,
                    prompt_lookup_num_tokens=10,
                )
            if args.debug:
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=Path, default=None)
    parser.add_argument("--data_dir", type=str, default='/raid/duan/cd58hofa/rainbow_bank')
    parser.add_argument("--dataset_name", type=str, default='quant_ph.txt')
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--vl_model", type=str, default='llama_3.2')
    parser.add_argument("--lookup_decoding", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args, left_argv = parser.parse_known_args()


    predictions = test(args)
