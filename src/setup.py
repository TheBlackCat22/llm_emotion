import os
import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

LABEL2ID =  {
    'sadness':0,
    'joy':1,
    'love':2,
    'anger':3,
    'fear':4,
}
def get_emotion_reward(split:str, cache_dir:str = None) -> Dataset:
    """Load the Emotion dataset from Huggingface and convert it into a mulitlabel dataset.
    """

    print(f'Creating Emotion-Reward Dataset {split} Split', flush=True)
    dataset = load_dataset('dair-ai/emotion', 'split', cache_dir=cache_dir)
    dataset = dataset.shuffle(seed=42)

    # Removing Surprise Label
    dataset = dataset.filter(lambda example: example['label']!=5)

    if split == 'train':
        dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    else:
        dataset = dataset['test']
    max_num_joins = 3

    data = {'text':[], 'sadness':[], 'joy':[], 'love':[], 'anger':[], 'fear':[]}
    start_idx = 0
    while True:

        print(f'Done {round(start_idx * 100 / len(dataset), 2)}%', end='\r')

        num_joins = int(torch.randint(low=1, high=max_num_joins+1, size=(1,)))
        end_idx = min(start_idx+num_joins, len(dataset))

        temp_dataset = dataset.select(range(start_idx, end_idx))

        joined_text = ' '.join(temp_dataset['text'])
        joined_label = set(temp_dataset['label'])

        data['text'].append(joined_text)
        for key in LABEL2ID.keys():
            if LABEL2ID[key] in joined_label:
                data[key].append(1)
            else:
                data[key].append(0)

        start_idx = end_idx

        if start_idx >=len(dataset):
            break

    data = Dataset.from_dict(data)
    return data


def get_emotion_sft(split:str, cache_dir:str=None) -> Dataset:
    """Load the Emotion dataset from Huggingface and convert it into a preference dataset for sft.
    """
    print(f'Creating Emotion-SFT Dataset {split} Split', flush=True)
    dataset = load_dataset('dair-ai/emotion', 'split', cache_dir=cache_dir)
    dataset = dataset.shuffle(seed=42)

    if split == 'train':
        dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    else:
        dataset = dataset['test']

    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large', cache_dir=cache_dir, clean_up_tokenization_spaces=True)

    data = {'prompt':[], 'chosen':[], 'rejected':[]}
    for row in tqdm(dataset, desc='Parsing Data'):

        tokens = tokenizer(row['text'], padding=False, truncation=False, add_special_tokens=False).input_ids

        if len(tokens) <= 4:
            continue

        prompt = tokenizer.decode(tokens[:4], skip_special_tokens=True)
        chosen = tokenizer.decode(tokens[4:], skip_special_tokens=True)
        rejected = None

        data['prompt'].append(prompt), data['chosen'].append(chosen), data['rejected'].append(rejected)

    data = Dataset.from_dict(data)
    return data


def main(config):

    print(load_dotenv())

    print('*************************') 
    print('Downloading Policy Models')
    print('*************************')

    for model_name in config['policy_models'].keys():

        model_path = os.path.join(config['model_dir'], model_name+'-policy')

        if not os.path.exists(model_path):
            print(f'Downloading {model_name}')
            model = AutoModelForCausalLM.from_pretrained(
                config['policy_models'][model_name], 
                cache_dir=config['cache_dir'], 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float32
                )
            tokenizer = AutoTokenizer.from_pretrained(
                config['policy_models'][model_name], 
                cache_dir=config['cache_dir'], 
                clean_up_tokenization_spaces=True, 
                padding_side='left', 
                truncation_side='right'
                )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id

            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            print(f'{model_name} saved to {model_path}\n')
        else:
            print(f'{model_name} already exists at {model_path}\n')

    
    print('*************************') 
    print('Downloading Reward Models')
    print('*************************')

    for model_name in config['reward_models'].keys():

        model_path = os.path.join(config['model_dir'], model_name+'-reward')

        if not os.path.exists(model_path):
            print(f'Downloading {model_name}')
            model = AutoModelForSequenceClassification.from_pretrained(
                config['reward_models'][model_name], 
                num_labels=len(LABEL2ID),
                cache_dir=config['cache_dir'], 
                torch_dtype=torch.float32,
                )
            tokenizer = AutoTokenizer.from_pretrained(
                config['reward_models'][model_name], 
                cache_dir=config['cache_dir'], 
                clean_up_tokenization_spaces=True, 
                padding_side='right', 
                truncation_side='right'
                )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id

            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            print(f'{model_name} saved to {model_path}\n')
        else:
            print(f'{model_name} already exists at {model_path}\n')


    print('*********************************')
    print('Downloading & Processing Datasets')
    print('*********************************')

    for dataset_name in config['datasets'].keys():
        
        dataset_path = os.path.join(config['dataset_dir'], dataset_name)
        
        if not os.path.exists(dataset_path):

            train_data = config['datasets'][dataset_name]('train', cache_dir=config['cache_dir'])
            train_data.to_json(os.path.join(dataset_path, 'train.jsonl'))

            test_data = config['datasets'][dataset_name]('test', cache_dir=config['cache_dir'])
            test_data.to_json(os.path.join(dataset_path, 'test.jsonl'))

            print(f'{dataset_name} saved to {dataset_path}\n')
        else:
            print(f'{dataset_name} already exists at {dataset_path}\n')


if __name__ == '__main__':

    config = {
        'policy_models' : {
            'gpt2' : 'openai-community/gpt2',
            'llama3.1' : 'meta-llama/Llama-3.1-8B'
        },
        'reward_models' : {
            'gpt2' : 'openai-community/gpt2'
        },
        'datasets' : {
            'emotion-reward' : get_emotion_reward,
            'emotion-sft' : get_emotion_sft
        },
        'cache_dir' : './.cache',
        'model_dir' : './models',
        'dataset_dir' : './datasets'
    }

    main(config)