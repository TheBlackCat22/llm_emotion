from tqdm import tqdm
import torch
from src.utils import(
    load_tokenizer,
    load_policy_model,
)


PolicyConfig = {
    'model_path': 'models/llama3.1-policy',
    'state_dict_path' : None,
    'quantization' : {
        'load_in_4bit':True,
        'bnb_4bit_quant_type':"nf4"
    },
    'lora': {
        'r':64,
        'lora_alpha':16,
        'lora_dropout':0.1,
        'bias':"none",
        'task_type':"CAUSAL_LM",
        'target_modules':"all-linear"
    } 
}
GenerationConfig = {
    'do_sample': True,
    'max_new_tokens': 64,
    'min_new_tokens': 1,
    'top_k': 0.0,
    'top_p': 1
}

prompts = ["i was feeling really", "i feel so thrilled", "i feel is that", "i tune out the", "i sit here writing"]


def sample_completions(dataset, tokenizer, model, generation_config, batch_size=64):

    # model.to('cuda')
    model.eval()

    generations = []
    for start_idx in tqdm(range(0, len(dataset), batch_size), desc='Generating Samples'):
        end_idx = min(len(dataset), start_idx+batch_size)

        batch = dataset[start_idx : end_idx]

        prompt_tokens = tokenizer(batch, padding=True, truncation=False, return_tensors='pt').to('cuda')
        gen_tokens = model.generate(**prompt_tokens, **generation_config, pad_token_id=tokenizer.pad_token_id)
        gen_texts = tokenizer.batch_decode(gen_tokens[:, prompt_tokens['input_ids'].shape[1]:], skip_special_tokens=True)
        generations.extend(gen_texts)

    return generations


policy_tokenizer = load_tokenizer(PolicyConfig['model_path'])
policy_model = load_policy_model(**PolicyConfig)

policy_model.print_trainable_parameters()
print(type(policy_model))
policy_model = policy_model.merge_and_unload()
# generations = sample_completions(prompts, policy_tokenizer, policy_model, GenerationConfig)
# print(generations)