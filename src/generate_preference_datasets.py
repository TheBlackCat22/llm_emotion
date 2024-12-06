import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear']


def create_id_dataset(data, key):

    def select_accepted_rejected(row):

        if row[f'{key}_score1']>row[f'{key}_score2']:
            chosen_idx = 1
            rejected_idx = 2
        else:
            chosen_idx = 2
            rejected_idx = 1
            
        new_row = {
            'chosen':row[f'sample{chosen_idx}'], 
            'rejected':row[f'sample{rejected_idx}']
        }

        for label in EMOTION_LABELS:
            new_row.update({
                f'chosen_{label}_score':row[f'{label}_score{chosen_idx}'],
                f'rejected_{label}_score':row[f'{label}_score{rejected_idx}'] 
            })

        return new_row
    
    columns_to_remove = data.column_names
    columns_to_remove.remove('prompt')

    data = data.map(select_accepted_rejected, desc=f'Creating {key} Indicator Dataset', remove_columns=columns_to_remove)
    
    return data


def calculate_rewards(data, model, tokenizer, batch_size):

    model.cuda()
    model.eval()

    def get_rewards(row):
        prompt_tokens = tokenizer(row['sample1']+row['sample2'], padding=True, truncation=False, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            gen_scores = torch.sigmoid(model(**prompt_tokens).logits)

        gen_scores1, gen_scores2 = gen_scores[:len(gen_scores)//2], gen_scores[len(gen_scores)//2:]

        new_row = {}
        for i, scores in enumerate([gen_scores1, gen_scores2]):
            for j, label in enumerate(EMOTION_LABELS):
                new_row[f'{label}_score{i+1}'] = scores[:, j]
        
        return new_row

    data = data.map(get_rewards, batched=True, batch_size=batch_size, desc='Calculating Rewards')
    return data


def sample_completion_pairs(data, model, tokenizer, generation_config, batch_size):

    model.to('cuda')
    model.eval()

    def generate(row):
        prompt_tokens = tokenizer(row['prompt'], padding=True, truncation=False, return_tensors='pt').to('cuda')
        gen_tokens = model.generate(**prompt_tokens, **generation_config, pad_token_id=tokenizer.pad_token_id, num_return_sequences=2)
        gen_texts = tokenizer.batch_decode(gen_tokens[:, prompt_tokens['input_ids'].shape[1]:], skip_special_tokens=True)

        gen_texts = np.array(gen_texts).reshape(-1, 2)

        return {'sample1':gen_texts[:,0], 'sample2':gen_texts[:,1]}

    data = data.map(generate, batched=True, batch_size=batch_size, desc='Sampling Completions Pairs', remove_columns=['chosen', 'rejected'])
    return data


def load_reward_model(model_path, state_dict_path=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, low_cpu_mem_usage=True)

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

    return model


def load_policy_model(model_path, state_dict_path=None):
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

    return model


def main(config):

    policy_tokenizer = AutoTokenizer.from_pretrained(config['policy_model_path'])
    policy_model = load_policy_model(config['policy_model_path'], config['policy_state_path'])

    reward_tokenizer = AutoTokenizer.from_pretrained(config['reward_model_path'])
    reward_model = load_reward_model(config['reward_model_path'], config['reward_state_path'])

    dataset = load_dataset('json', data_dir=config['dataset_path'])

    for split in ['test', 'train']:
        print('\n\n************************')
        print(f'Processing {split} split')
        print('************************')
        
        data = dataset[split]
        
        data = sample_completion_pairs(data, policy_model, policy_tokenizer, config['generation_config'], config['batch_size'])
    
        data = calculate_rewards(data, reward_model, reward_tokenizer, config['batch_size'])

        for label in EMOTION_LABELS:
            create_id_dataset(data, label).to_json(os.path.join('datasets', f'emotion-preference-{label}', split + '.jsonl'))


if __name__ == '__main__':

    config = {
        'dataset_path' : 'datasets/emotion-sft',
        'policy_model_path' : 'models/gpt2-policy',
        'policy_state_path' : 'outputs/sft/model.pt',
        'reward_model_path' : 'models/gpt2-reward',
        'reward_state_path' : 'outputs/reward/model.pt',
        'batch_size' : 64,
        'generation_config' : {
            'do_sample': True,
            'max_new_tokens': 64,
            'min_new_tokens': 1,
            'top_k': 0.0,
            'top_p': 1
        }
    }
    
    main(config)