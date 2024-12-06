import os
import torch
from tqdm import tqdm
from pprint import pprint
from datasets import load_dataset

from src.utils import(
    print_item,
    load_tokenizer,
    load_policy_model,
    load_reward_model
)


def sample_completions(dataset, tokenizer, model, generation_config, batch_size=64):

    model.to('cuda')
    model.eval()

    generations = []
    for start_idx in tqdm(range(0, len(dataset), batch_size), desc='Generating Samples'):
        end_idx = min(len(dataset), start_idx+batch_size)

        batch = dataset[start_idx : end_idx]

        prompt_tokens = tokenizer(batch['prompt'], padding=True, truncation=False, return_tensors='pt').to('cuda')
        gen_tokens = model.generate(**prompt_tokens, **generation_config, pad_token_id=tokenizer.pad_token_id)
        gen_texts = tokenizer.batch_decode(gen_tokens[:, prompt_tokens['input_ids'].shape[1]:], skip_special_tokens=True)
        generations.extend(gen_texts)

    return generations


def calculate_rewards(generations, tokenizer, model, batch_size=96):

    model.to('cuda')
    model.eval()

    rewards = []
    for start_idx in tqdm(range(0, len(generations), batch_size), desc='Calculating Rewards'):
        end_idx = min(len(generations), start_idx+batch_size)

        batch = generations[start_idx : end_idx]

        tokens = tokenizer(batch, padding=True, truncation=False, return_tensors='pt').to('cuda')
        with torch.no_grad():
            logits = model(**tokens).logits
        scores = torch.sigmoid(logits)

        rewards.extend(scores.tolist())

    rewards = torch.tensor(rewards).mean(dim=0).tolist()
    return rewards


def main(config):

    policy_tokenizer = load_tokenizer(config['PolicyConfig']['model_path'])
    policy_model = load_policy_model(**config['PolicyConfig'])
    print_item(f"Loaded {config['PolicyConfig']['model_path'].split('/')[-1]} Policy with weights from {config['PolicyConfig'].get('state_dict_path')}")

    reward_tokenizer = load_tokenizer(config['RewardConfig']['model_path'])
    reward_model = load_reward_model(**config['RewardConfig'])
    print_item(f"Loaded {config['RewardConfig']['model_path'].split('/')[-1]} Reward Model with weights from {config['RewardConfig'].get('state_dict_path')}")

    dataset = load_dataset('json', data_dir=config['DatasetConfig']['dataset_path'])

    generations = sample_completions(dataset['test'], policy_tokenizer, policy_model, config['GenerationConfig'])
    rewards = calculate_rewards(generations, reward_tokenizer, reward_model)

    print(rewards)


if __name__ == '__main__':

    config = {
        'DatasetConfig': {
            'dataset_path': 'datasets/emotion-sft'
        },  
        'PolicyConfig' : {
            'model_path': 'models/llama3.1-policy',
            'state_dict_path' : None,
            'quantization' : {
                'load_in_4bit':True,
                'bnb_4bit_quant_type':"nf4"
            }
        },
        'RewardConfig' : {
            'model_path': 'models/gpt2-reward',
            'state_dict_path' : 'outputs/reward/model.pt',
        },
        'GenerationConfig' : {
            'do_sample': True,
            'max_new_tokens': 64,
            'min_new_tokens': 1,
            'top_k': 0.0,
            'top_p': 1
        }
    }
    
    main(config)