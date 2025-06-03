import argparse
import logging
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from transformers import BertConfig, BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from asmtransformers.models.asmbert import ASMBertForMaskedLM


def pretrain(model_path, output_dir, data, tokenizer, epoch, batch_size, gradient_accumulation_steps,
             save_steps, logging_steps, mlm_prob):
    output_dir = f'{output_dir}/pretraining_mlm_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(filename=f"{output_dir}/training_logging.log")])

    logging.info("Script used: pretraining with MLM")
    logging.info(f"Saving checkpoints to: {output_dir}")

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    if model_path:
        model = ASMBertForMaskedLM.from_pretrained(model_path)
        assert model.base_model.embeddings.position_embeddings is model.base_model.embeddings.word_embeddings
    else:
        # if no model_path is given, initialise a 'clean ASMBert'
        config = BertConfig.from_json_file("asmtransformers/models/arm64bert/arm64bert_config.json")
        model = ASMBertForMaskedLM(config)
        assert model.base_model.embeddings.position_embeddings is model.base_model.embeddings.word_embeddings
    logging.info("Tokenizer and model loaded")

    # Load the training and evaluation datasets
    functions = load_from_disk(data)
    train_tokenized = functions['train'].shuffle(seed=42)
    eval_tokenized = functions['test'].shuffle(seed=42).select(range(100_000))
    logging.info("Datasets loaded")

    # given our tokenization, there's no sense in differentiating between whole word and token masking
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.06,
        num_train_epochs=epoch,
        evaluation_strategy="steps" if eval_tokenized is not None else "no",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_steps=save_steps // 3,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=5,
        prediction_loss_only=True,
        report_to=['tensorboard'],
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
    )

    logging.info(f"Save tokenizer to: {output_dir}")
    tokenizer.save_pretrained(output_dir)

    logging.info("Start training")
    logging.info(f"Model parameters: epochs={epoch}, eval_steps={save_steps}, batch_size={batch_size},"
                 f"mlm_prob={mlm_prob}")

    trainer.train()

    logging.info(f"Save model to: {output_dir}")
    model.save_pretrained(output_dir)

    logging.info("Training done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ASM-Pretrain")
    parser.add_argument("--model-path", type=str, default=None,
                        help='the path of the model to pretrain, can be empty if you want to initialise a new model')
    parser.add_argument("--output-dir", type=str, default=Path(__file__) / '../output',
                        help='the directory where the pretrained model be saved')
    parser.add_argument("--data", type=str,
                        help='training dataset')
    parser.add_argument("--tokenizer", type=str, default=Path(__file__) / '../asmtransformers/models/arm64bert',
                        help='the path of tokenizer')
    parser.add_argument("--epoch", type=int, default=1, help='number of training epochs')
    parser.add_argument("--batch-size", type=int, default=1, help='training batch size')
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help='gradient accumulation steps')
    parser.add_argument("--save-steps", type=int, default=10000, help='after how many steps evaluate and save model')
    parser.add_argument("--logging-steps", type=int, default=100, help='number of update steps between two logs')
    parser.add_argument("--mlm-prob", type=int, default=0.15, help='probability of a token/word to be masked')

    args = parser.parse_args()

    pretrain(**vars(args))
