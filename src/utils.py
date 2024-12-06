import os
import yaml
import torch
import shutil
import pprint
import argparse
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Config File')
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    return parser.parse_args()


def parse_config(args):

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.model_path is not None:
        config['ModelConfig']['model_path'] = args.model_path
    
    if args.dataset_path is not None:
        config['DatasetConfig']['dataset_path'] = args.dataset_path
    
    if args.output_dir is not None:
        config['OutputDir'] = args.output_dir

    os.makedirs(config['OutputDir'], exist_ok=True)
    with open(os.path.join(config['OutputDir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    return config


def print_heading(heading):
    print_item("\n\n" + "*"*len(heading))
    print_item(heading)
    print_item("*"*len(heading))


def print_item(item):
    if isinstance(item, dict):
        print(pprint.pformat(item, indent=4, width=2), flush=True)
    else:
        print(item, flush=True)


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def load_reward_model(model_path, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, low_cpu_mem_usage=True)

    if kwargs.get('state_dict_path') is not None:
        state_dict = torch.load(kwargs.get('state_dict_path'), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

    return model


def load_policy_model(model_path, ref=False, **kwargs):

    state_dict_path = kwargs.pop('state_dict_path', None)
    quantization_config = kwargs.pop('quantization', None)
    lora_config = kwargs.pop('lora', None)

    if quantization_config is not None:
        quantization_config = BitsAndBytesConfig(**quantization_config, bnb_4bit_compute_dtype=torch.float16)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, quantization_config=quantization_config)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

    if (lora_config is not None) and (ref is False):
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)

    return model


def parse_reward_data(dataset, tokenizer):

    def process_row(row):
        row = dict(row)
        new_row = tokenizer(row.pop('text'), padding=False)
        new_row['labels'] = list(row.values())
        return new_row

    dataset = dataset.map(process_row, desc='Parsing Data', remove_columns=dataset['train'].column_names)
    return dataset


def parse_sft_data(dataset):

    def process_row(row):
        return {'text':row['prompt']+row['chosen']}

    dataset = dataset.map(process_row, desc='Parsing Data', remove_columns=dataset['train'].column_names)
    return dataset


def compute_reward_loss(outputs, labels, num_items_in_batch):
    logits = outputs.get("logits")
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))
    return loss


def compute_reward_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()

    metrics = {}
    metrics['accuracy'] = accuracy_score(labels.astype(int), (predictions > 0.5).astype(int))
    metrics['roc_auc'] = roc_auc_score(labels.astype(int), predictions, average='macro')

    return metrics


def save_model_dict(model, output_dir):
    if isinstance(model, PeftModelForCausalLM):
        model = model.merge_and_unload()
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))


def delete_checkpoints(output_dir):
    for folder_name in os.listdir(output_dir):
        
        folder_path = os.path.join(output_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        if not folder_name.startswith('checkpoint'):
            continue

        shutil.rmtree(folder_path)