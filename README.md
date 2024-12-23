# PromptLookup & CopyLookup Decoding for End-to-End PDF to Markdown conversion

Please use the modified version of transformers package `pip install git+https://github.com/tic-top/transformers.git` before [this PR](https://github.com/huggingface/transformers/pull/31711) has been merged into the main repo, because of kosmos-2.5 support.

Dataset is annotated by [LaTeX Rainbow](https://github.com/InsightsNet/texannotate) and tentatively named as RainbowBank.
In the `data` folder you can find example data, and a list of paper ids that need to be downloaded from arXiv for reproduction of the results.

## Repository Structure
```
root
├── data
│   ├── rainbow_bank           >>  sample data and arXiv ids to build the complete dataset
│   ├── data_maker.py          >>  make dataset from output of LaTeX Rainbow
│   └── data_spliter.py        >>  split dataset
├── ernie_ckpt                 >>  fine-tuned Ernie Layout LoRA weight
├── ernie_layout_pytorch       >>  implementation of Ernie Layout model in PyTorch
├── nougat                     >>  implementation of Nougat model
├── mugat                      >>  implementation of Mugat model
├── token-path-prediction      >>  implementation of dataloader
├── patching_*.py              >>  Injecting code into HuggingFace
├── tokenizer*                 >>  Patching Ernie Layout's tokenizer to fix its bug
├── training_ernie.py          >>  fine-tuning code for Ernie Layout
├── requirements.txt
├── test_markdown.py           >>  get scores for the baseline, PromptLookup, CopyLookup
└── utils.py
```

# More models on the waiting list
- [Vary](https://github.com/Ucas-HaoranWei/Vary) (Weights not released yet, and emails requesting weights were never answered)

## Acknowledgments
This repository builds on top of the [Nougat](https://github.com/facebookresearch/nougat), [Mugat](https://github.com/aimagelab/mugat?tab=readme-ov-file), [ERNIE-Layout-Pytorch](https://github.com/NormXU/ERNIE-Layout-Pytorch?tab=readme-ov-file) and [Token-Path-Prediction](https://github.com/WinterShiver/Token-Path-Prediction) repository.