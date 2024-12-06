from datasets import load_dataset
from trl import (
    SFTConfig, 
    SFTTrainer, 
    DPOConfig, 
    DPOTrainer
)
from transformers import (
    DataCollatorWithPadding, 
    TrainingArguments, 
    Trainer
)

from src.utils import (
    parse_args,
    parse_config,
    print_heading,
    print_item,
    load_tokenizer,
    load_reward_model,
    load_policy_model,
    parse_reward_data,
    parse_sft_data,
    compute_reward_loss, 
    compute_reward_metrics,
    save_model_dict,
    delete_checkpoints
)


def main(args):

    print_heading('Config')
    config = parse_config(args)
    print_item(config)


    print_heading('Models')
    model_name = config['ModelConfig']['model_path'].split('/')[-1]
    tokenizer = load_tokenizer(config['ModelConfig']['model_path'])
    print_item(f"Loaded {model_name} Tokenizer")
    if config['Task'] == 'reward':
        model = load_reward_model(**config['ModelConfig'])
        print_item(f"Loaded {model_name} Reward Model with weights from {config['ModelConfig'].get('state_dict_path')}")
    else:
        model = load_policy_model(**config['ModelConfig'])
        print_item(f"Loaded {model_name} Policy with weights from {config['ModelConfig'].get('state_dict_path')}")
        if config['Task'] == 'dpo':
            ref_model = load_policy_model(**config['ModelConfig'], ref=True)
            print_item(f"Loaded {model_name} Reference Policy with weights from {config['ModelConfig'].get('state_dict_path')}")


    print_heading('Dataset')
    print_item(f"Loading {config['DatasetConfig']['dataset_path'].split('/')[-1]} Dataset")
    dataset = load_dataset('json', data_dir=config['DatasetConfig']['dataset_path'])
    print_item(dataset) 
    if config['Task'] == 'reward':
        dataset = parse_reward_data(dataset, tokenizer)
    elif config['Task'] == 'sft':
        dataset = parse_sft_data(dataset)

    
    print_heading('Training')
    common_args = {
        'output_dir':config['OutputDir'],
        'overwrite_output_dir':True,
        'eval_strategy':'steps',
        'logging_dir':config['OutputDir'],
        'save_only_model':True,
        'load_best_model_at_end':True,
        'metric_for_best_model':'eval_loss',
        'seed':config['Seed'],
        'report_to':'tensorboard',
        'save_total_limit':1,
        'eval_on_start':True,
        'logging_steps':1
    }
    if config['Task'] == 'reward':
        training_args = TrainingArguments(
            **common_args,
            **config['TrainerConfig']
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
            compute_loss_func=compute_reward_loss,
            compute_metrics=compute_reward_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )
    elif config['Task'] == 'sft':
        training_args = SFTConfig(
            **common_args,
            **config['TrainerConfig']
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
        )
    elif config['Task'] == 'dpo':
        training_args = DPOConfig(
            **common_args,
            **config['TrainerConfig']
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
        )
    
    trainer.train()
    save_model_dict(trainer.model, config['OutputDir'])

    delete_checkpoints(config['OutputDir'])



if __name__ == '__main__':
    args = parse_args()
    main(args)