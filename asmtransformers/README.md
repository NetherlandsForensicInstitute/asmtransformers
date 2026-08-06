ASM Transformers
================

Binary code similarity models using Transformers. Pronounced: _awesome transformers_.
The actual model can be found [on Hugging Face](https://huggingface.co/NetherlandsForensicInstitute/Multilingual-ASMBERT)

Background
----------
Inspired by [jTrans](https://github.com/vul337/jTrans), which implements a _jump-aware_ BERT-model for x86-assembly code
similarity.
For details on jTrans see Wang, Hao, et al. "JTrans: Jump-Aware Transformer for Binary Code Similarity Detection" 
_Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis_. 2022.

This project started focussed on implementing the concepts from jTrans for ARM-assembly code in a clean and concise way
(see release v1.0.0). The current version of this repo consists of the code to train, finetune and evaluate a 
multilingual model of amd64, arm64, i386 and riscv64 assembly code.

Applications
------------
Binary code similary models can be used for _semantic code search_,
just as sentence embedding models for natural text can be used for [semantic text search](https://www.sbert.net/examples/applications/semantic-search/README.html).
Semantic code search can be useful when reverse engineering binary code and wanting to identify the purpose of an unknown function.
The unknown function's assembly code can be embedded using a binary code similarity model and compared to a database of known functions.


Dataset
-----

The dataset is built on the official [Debian Repository](https://wiki.debian.org/DebianRepository). To obtain multiple families of
assembly, we used `apt` to cross-build the same source package to multiple architectures. The idea is that this gives us 
the same functions for all four architectures. For all four architectures, these functions are compiled with different optimisation:
O0, O1, O2, O3, Os and manually selected set with advanced instructions further referenced here as Oc for Optimised-Custom.
This results in a maximum of 24 (6 optimisation * 4 architectures) different functions 
which are semantically similar. (i.e. they represent the same source code but are compiled differently)
In practise, it was much easier to obtain amd64 functions than riscv64 functions. Thus, not all functions have 24 semantically similar functions.

The dataset is split into a train, test and an evaluation set. This in done on source package, so all binaries and functions belonging to one source package are part of
either the train or the test set, not both.

### Total amount of functions per architecture

| Architecture | # functions|
|--------------|------------|
| amd64        |  8 202 164 |
| i386	       |  4 868 531 |
| arm64	       |  4 421 768 |
| riscv64	   |  3 791 434 |
-----------------------------


Pipeline
--------
With a dataset as described above, we train a BERT model using Masked Language Modelling (Devlin et al., 2019) and Jump
Target Prediction (Wang et al., 2022). The result is a BERT model that "speaks" different varieties of assembly. The next step is to teach
the model which pieces of code are similar, and which ones are not. This is a key step in any [semantic search](https://sbert.net/index.html)
model. The model sees triplets: two functions that have been compiled in different ways (i.e. code that works the same, but looks
different) and one completely different function. We teach the model that the anchor and positive example look alike,
whereas the anchor and the negative example do not by means of triplet loss.

The result is a model that can encode binary code in an embedding. A database of known functions is created by embedding
all functions. Then, a new, unknown function is encoded and compared to the database. The known functions are ranked by
their similarity to the unknown function, hopefully giving an indication of what this function does.

Pretraining
-----------
For cluster-oriented multi-architecture pretraining with CUDA bf16 mixed precision, see
[docs/pretraining.md](docs/pretraining.md). Run `scripts/pretrain.py --help` for the full current CLI.

    usage: pretrain.py [-h] [--model-path MODEL_PATH] [--data DATA] [--tokenizer TOKENIZER] [--config CONFIG]
                   [--epoch EPOCH] [--max-steps MAX_STEPS] [--batch-size BATCH_SIZE]
                   [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] [--save-steps SAVE_STEPS]
                   [--logging-steps LOGGING_STEPS] [--mlm-prob MLM_PROB] [--learning-rate LEARNING_RATE]
                   [--warmup-steps WARMUP_STEPS] [--bf16] [--tf32] [--dataloader-num-workers DATALOADER_NUM_WORKERS]
                   [--save-total-limit SAVE_TOTAL_LIMIT] [--eval-samples EVAL_SAMPLES] [--seed SEED]
                   [--resume-from-checkpoint RESUME_FROM_CHECKPOINT] [--run-id RUN_ID]
                   output_dir


    ASM-Pretrain
    
    positional arguments:
      output_dir            the directory where the pretrained model will be saved
    
    options:
      -h, --help            show this help message and exit
      --model-path MODEL_PATH
                            the path of the model to pretrain, can be empty if you want to initialise a new model
      --data DATA           training dataset
      --tokenizer TOKENIZER
                            the path of tokenizer; defaults to the packaged multilingual_asmbert tokenizer
      --config CONFIG       the path of the model config used when initializing a new model. Defaults to packaged multilingual_asmbert
      --epoch EPOCH         number of training epochs
      --max-steps MAX_STEPS
                            maximum number of training steps; -1 uses epochs
      --batch-size BATCH_SIZE
                            training batch size
      --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                            gradient accumulation steps
      --save-steps SAVE_STEPS
                            after how many steps evaluate and save model
      --logging-steps LOGGING_STEPS
                            number of update steps between two logs
      --mlm-prob MLM_PROB   probability of a token/word to be masked
      --learning-rate LEARNING_RATE
                            learning rate
      --warmup-steps WARMUP_STEPS
                            warmup steps for the learning-rate scheduler
      --bf16                enable CUDA bfloat16 mixed precision training
      --tf32                enable TF32 matmul/cudnn on supported CUDA GPUs
      --dataloader-num-workers DATALOADER_NUM_WORKERS
                            number of worker processes used by each training dataloader
      --save-total-limit SAVE_TOTAL_LIMIT
                            maximum number of checkpoints to keep
      --eval-samples EVAL_SAMPLES
                            maximum number of test samples used for intermediate evaluation; use -1 to disable the limit
      --seed SEED           training seed
      --resume-from-checkpoint RESUME_FROM_CHECKPOINT
                            path to a Trainer checkpoint to resume from
      --run-id RUN_ID       stable run id used under output_dir; overrides ASMTRANSFORMERS_RUN_ID, SLURM_JOB_ID, and timestamp

We take the tokenized binaries (preferably in the shape of arrow files, but anything that can be called with the huggingface
datasets load_from_disk function works). If no model path is given, we initialise a model from scratch. Otherwise, this
code will continue training your model. Maximum 100.000 functions from the test set are used for intermediate evaluation,
for speed purposes. Then, Masked Language Modelling (MLM) is performed. The Jump Target Prediction task, as proposed
by Wang et al. in the jTrans paper (referred to above), is implicitly included in the MLM procedure. In case of a masked
JUMP-token, the correct token to predict is the correct jump address. This is the token index of the place the code was
supposed to jump to, materialised in the vocabulary as JUMP_ADDR_n `(n = 1, len(max_token_lenght))`.

You can also find our monolingual, pretrained only ARM64BERT model on Huggingface:
<a href='https://huggingface.co/NetherlandsForensicInstitute/ARM64Bert'>NetherlandsForensicInstitute/ARM64Bert</a>

Finetuning
----------

    usage: scripts/finetune.py [-h] [-b BATCH_SIZE] data_folder model

    positional arguments:
      data_folder           folder with input data
      model                 The name of the model used for finetuning
    
    options:
      -h, --help            show this help message and exit
      -b, --batch-size BATCH_SIZE
                            Feed the data to the model in batches for a potential speed-up

The finetune code will take the data and turn it into "triplets": it takes one function that has been compiled in two
different ways. These are the anchor and positive example (similar to [Sentence BERT](https://sbert.net/docs/sentence_transformer/dataset_overview.html)).
Then, 2 negative examples are randomly sampled from the data. These triplets are passed to the model and the model is
trained such that the anchor and the positive example are closer to each other in embedding space (e.g. by measuring
Cosine distance) than the anchor and the negative examples. We use [BatchSemiHardTripletLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#batchsemihardtripletloss)
to train the model.

Our multilingual model can be found on [Hugging Face](https://huggingface.co/NetherlandsForensicInstitute/Multilingual-ASMBERT)

You can also find our monolingual, ARM64BERT semantic search model on Hugging Face:
<a href='https://huggingface.co/NetherlandsForensicInstitute/ARM64bert-embedding'>NetherlandsForensicInstitute/ARM64bert-embedding</a>

Evaluation
----------

    usage: scripts/evaluation.py [-h] [--pool-size POOL_SIZE] [--seed SEED] [--repeats REPEATS] [--static-pool] input_path output_path

    evaluation
    
    positional arguments:
      input_path            the path to the anchors/positives/negative pools
      output_path           the path to write the final scores to
    
    options:
      -h, --help            show this help message and exit
      --pool-size POOL_SIZE
                            the poolsize to pick the positive example from
      --seed SEED           seed random evaluation sampling
      --repeats REPEATS     number of static-pool evaluation repeats
      --static-pool         keep the negatives pool or refresh for every anchor-pos pair

Keep in mind that the pool-size-parameter does not include the positive example. For example if we want to conduct the
experiment with pool-size 32, we need a pool of 31 negatives and 1 positive example. Therefore the input of the pool-
size parameter is 31.

Pass `--seed` to make anchor/positive and negative sampling reproducible. If omitted, evaluation keeps using the
current unseeded random sampling behavior.

Pass `--repeats` with `--static-pool` to reuse one sampled anchor/positive set while varying only the negative pool.
Repeated runs write an aggregate CSV with per-repeat final MRR/P@1 plus mean, standard deviation, minimum, and maximum.
Repeats greater than 1 are not supported for dynamic pools.

The performance of the models is evaluated according to the methods in jTrans. For this evaluation we create triplets of
any chosen function (which we call the anchor); the same function on a different compilation level (the positive
example); and a pool of either 31 or 10.000 other functions (the negative examples).

We calculate the cosine similarity between the anchor and the positive example; and the cosine similarities between the
anchor and each of the negative examples. We rank these cosine similarities and calculate the Mean Reciprocal Rank and
Recall@1 for the positive example.

There are a few minor things that we do differently than jTrans. Firstly, they do not check if the input of the
positive example is equal to the input of the anchor. Especially for ARM64, there seem to be a significant number of
cases where different optimisation levels return the same output. This seems like it would unfairly inflate the scores,
so we make sure that the input of the positive example is never the same as the input of the anchor. We also
check for duplicate anchor-positive pairs.

Additionally, we make sure that none of the negative examples have the same input as the anchor. As there are possibly
duplicate functions in our dataset, we want to avoid the possibility that a function in the list of negative examples
is actually the same as the positive example as this would result in a false negative.

Finally, we found that evaluation metrics of the same model differed hugely when evaluation was run at different seeds.
Therefore, we created one evaluation dataset (i.e. anchor-positives-negatives) that was used for the evaluation of all
models we trained. This dataset was created using eval-datasets.py. A separate evaluation file, namely 
evaluation-static-dataset.py, was used to evaluate this dataset, in order to not overcomplicate evaluation.py.

Inference
---------

    usage: scripts/evaluation.py [-h] [--pool-size POOL_SIZE] [--seed SEED] [--repeats REPEATS] [--static-pool] input_path output_path

    evaluation
    
    positional arguments:
      input_path            the path to the anchors/positives/negative pools
      output_path           the path to write the final scores to
    
    options:
      -h, --help            show this help message and exit
      --pool-size POOL_SIZE
                            the poolsize to pick the positive example from
      --seed SEED           seed random evaluation sampling
      --repeats REPEATS     number of static-pool evaluation repeats
      --static-pool         keep the negatives pool or refresh for every anchor-pos pair

inference.py adds a column to the given dataset, called 'embeddings', containing the embeddings
corresponding to each function, and writes it to the output folder.

Ghidra plug-in
--------------
The plugin to use this model in Ghidra, [Sententia](../sententia), is available in this repository.

Prerequisites
-------------
Python 3.13

Requirements
------------

Installing this project locally can be done using `pip`:

```
$ python3 -m pip install .
```

For further development, this project uses [PDM](https://pdm-project.org/en/latest/) and `pyproject.toml` to manage dependencies.
See [PDM's installation instructions](https://pdm-project.org/en/latest/#installation) to get started,
and subsequently call `pdm install` from the project's directory to automatically create a new virtual environment with dependencies.
