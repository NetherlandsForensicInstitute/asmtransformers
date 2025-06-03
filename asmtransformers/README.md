ASM Transformers
================

Binary code similarity models using Transformers. Pronounced: _awesome transformers_.

Background
----------
Inspired by [jTrans](https://github.com/vul337/jTrans), which implements a _jump-aware_ BERT-model for x86-assembly code 
similarity.
For details on jTrans see Wang, Hao, et al. "Jtrans: Jump-aware transformer for binary code similarity detection." _Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis_. 2022.
 
For now, we focus on implementing the concepts from jTrans for ARM-assembly code in a clean and concise way. 
For future work we hope to train a model on an intermediate representation, in order to create a cross-architecture model. 

Applications
------------
Binary code similary models can be used for _semantic code search_, 
just as sentence embedding models for natural text can be used for [semantic text search](https://www.sbert.net/examples/applications/semantic-search/README.html).
Semantic code search can be useful when reverse engineering binary code and wanting to identify the purpose of an unknown function. 
The unknown function's assembly code can be embedded using a binary code similarity model and compared to a database of known functions.


Dataset
-----
The dataset is created in the same way as Wang et al. create Binary Corp. A large set of binary code comes from the 
[ArchLinux official repositories](https://aur.archlinux.org/) and the [ArchLinux user repositories](https://archlinux.org/packages/).
All this code is split into functions that are compiled with different optimisation 
(O0, O1, O2, O3 and O3) and security settings (fortify or no-fortify). This results
in a maximum of 10 (5*2) different functions which are semantically similar i.e. they represent the same functionality but are written differently. 
The dataset is split into a train and a test set. This in done on project level, so all binaries and functions belonging to one project are part of 
either the train or the test set, not both. We have not performed any deduplication on the dataset for training.

| set   | # functions |
|-------|------------:|
| train |  18,083,285 |
| test  |   3,375,741 |

Pipeline
--------
With a dataset as described above, we train a BERT model using Masked Language Modelling (Devlin et al., 2019) and Jump 
Target Prediction (Wang et al., 2022). The result is a BERT model that "speaks" ARM64 assembly. The next step is to teach 
the model which pieces of code are similar, and which ones are not. This is a key step in any [semantic search](https://sbert.net/index.html)
model. The model sees triplets: two functions that have been compiled in different ways (i.e. code that works the same, but looks 
different) and one completely different function. We teach the model that the anchor and positive example look alike, 
whereas the anchor and the negative example do not by means of triplet loss. 

The result is a model that can encode binary code in an embedding. A database of known functions is created by embedding
all functions. Then, a new, unknown function is encoded and compared to the database. The known functions are ranked by 
their similarity to the unknown function, hopefully giving an indication of what this function does.

Pretraining
-----------

    usage: scripts/pretrain.py [-h] [--model-path MODEL_PATH] [--output-dir OUTPUT_DIR] [--data DATA] [--tokenizer TOKENIZER] [--epoch EPOCH] [--batch-size BATCH_SIZE] [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] [--save-steps SAVE_STEPS] [--logging-steps LOGGING_STEPS] [--mlm-prob MLM_PROB]
    
    ASM-Pretrain
    
    options:
      -h, --help            show this help message and exit
      --model-path MODEL_PATH
                            the path of the model to pretrain, can be empty if you want to initialise a new model
      --output-dir OUTPUT_DIR
                            the directory where the pretrained model be saved
      --data DATA           training dataset
      --tokenizer TOKENIZER
                            the path of tokenizer
      --epoch EPOCH         number of training epochs
      --batch-size BATCH_SIZE
                            training batch size
      --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                            gradient accumulation steps
      --save-steps SAVE_STEPS
                            after how many steps evaluate and save model
      --logging-steps LOGGING_STEPS
                            number of update steps between two logs
      --mlm-prob MLM_PROB   probability of a token/word to be masked

We take the tokenized binaries (preferably in the shape of arrow files, but anything that can be called with the huggingface 
datasets load_from_disk function works). If no model path is given, we initialise a model from scratch. Otherwise, this
code will continue training your model. Maximum 100.000 functions from the test set are used for intermediate evaluation,
for speed purposes. Then, Masked Language Modelling (MLM) is performed. The Jump Target Prediction task, as proposed
by Wang et al. in the jTrans paper (referred to above), is implicitly included in the MLM procedure. In case of a masked
JUMP-token, the correct token to predict is the correct jump address. This is the token index of the place the code was
supposed to jump to, materialised in the vocabulary as JUMP_ADDR_n `(n = 1, len(max_token_lenght))`.

The resulting model, 
<a href='https://huggingface.co/NetherlandsForensicInstitute/ARM64Bert'>NetherlandsForensicInstitute/ARM64Bert</a> 
is available on Huggingface Hub.

Finetuning
----------

    usage: scripts/finetune.py [-h] -d DATA_FOLDER -m MODEL [-b BATCH_SIZE]
    
    options:
      -h, --help            show this help message and exit
      -d DATA_FOLDER, --data-folder DATA_FOLDER
                            folder with data
      -m MODEL, --model MODEL
                            The name of the model used for finetuning
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Feed the data to the model in batches for a potential speed-up

The finetune code will take the data and turn it into "triplets": it takes one function that has been compiled in two 
different ways. These are the anchor and positive example (similar to [Sentence BERT](https://sbert.net/docs/sentence_transformer/dataset_overview.html)).
Then, 2 negative examples are randomly sampled from the data. These triplets are passed to the model and the model is 
trained such that the anchor and the positive example are closer to each other in embedding space (e.g. by measuring 
Cosine distance) than the anchor and the negative examples. We use [BatchSemiHardTripletLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#batchsemihardtripletloss)
to train the model.

The resulting model, 
<a href='https://huggingface.co/NetherlandsForensicInstitute/ARM64bert-embedding'>NetherlandsForensicInstitute/ARM64bert-embedding</a> 
is available on Huggingface Hub.

Evaluation
----------

    usage: scripts/evaluation.py [-h] [--input-path INPUT_PATH] [--output-path OUTPUT_PATH] [--pool-size POOL_SIZE] [--static-pool]

    evaluation

    options:
      -h, --help            show this help message and exit
      --input-path INPUT_PATH
                        the path to the test data
      --output-path OUTPUT_PATH
                        the path to write the final scores to
      --pool-size POOL_SIZE
                        the poolsize to pick the positive example from
      --static-pool         keep the negatives pool or refresh for every anchor-pos pair

Keep in mind that the pool-size-parameter does not include the positive example. For example if we want to conduct the 
experiment with pool-size 32, we need a pool of 31 negatives and 1 positive example. Therefore the input of the pool-
size parameter is 31.

The performance of the models is evaluated according to the methods in jTrans. For this evaluation we create triplets of
any chosen function (which we call the anchor); the same function on a different compilation level (the positive 
example); and a pool of either 31 or 10.000 other functions (the negative examples). 

We calculate the cosine similarity between the anchor and the positive example; and the cosine similarities between the
anchor and each of the negative examples. We rank these cosine similarities and calculate the Mean Reciprocal Rank and
Recall@1 for the positive example. 

There are a few minor things that we do differently than jTrans. Firstly, they do not check if the input of the 
positive example is equal to the input of the anchor. Especially for ARM64, there seem to be a significant number of 
cases where different optimisation levels return the same output. This seems like it would unfairly inflate the scores,
so we make sure that the input of the positive example is never the same as the input of the anchor. 

Additionally, we make sure that none of the negative examples have the same input as the anchor. As there are possibly 
duplicate functions in our dataset, we want to avoid the possibility that a function in the list of negative examples
is actually the same as the positive example as this would result in a false negative. 

Inference
---------

    scripts/inference.py -d DATA_FOLDER -o OUTPUT_FOLDER -m MODEL_PATH

    options:
      -h, --help            show this help message and exit
      -d DATA_FOLDER, --data-folder DATA_FOLDER
                        folder containing to be inferenced data
      -o OUTPUT_FOLDER, --output-folder OUTPUT_FOLDER
                        folder to save embeddings
      -m MODEL_PATH, --model-path MODEL_PATH
                        (path to) model that should be used for inference

inference.py adds a column to the given dataset, called 'embeddings', containing the embeddings 
corresponding to each function, and writes it to the output folder.

Ghidra plug-in
--------------
The plugin to use this model in Ghidra will be made available.

Prerequisites
-------------
Python 3.12 or newer.

Requirements
------------

Installing this project locally can be done using `pip`:

```
$ python3 -m pip install .
```

For further development, this project uses [PDM](https://pdm-project.org/en/latest/) and `pyproject.toml` to manage dependencies.
See [PDM's installation instructions](https://pdm-project.org/en/latest/#installation) to get started, 
and subsequently call `pdm install` from the project's directory to automatically create a new virtual environment with dependencies.
