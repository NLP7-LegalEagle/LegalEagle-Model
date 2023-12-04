import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoPeftModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel
import re

import pandas as pd

def model_validation(name, model, validation_path):
    df = pd.read_csv(validation_path)
    result = pd.DataFrame()
    result['prompt'] = df['input'][:100]
    result['example'] =  df['output'][:100]
    result[name] = result['prompt'].apply(lambda prompt: model.text_generation(prompt))
    result.to_csv(f"{name}_validation_result.csv")
    print("----- finished -------")


dataset_path = "./datasets/dataset_validation_shuffle.csv"

model_validation("model1", model1, dataset_path)
