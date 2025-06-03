ASMTransformers 🦾
==================

ASMTransformers is a project to train and use a machine learning model to compare assembly (currently only ARM64)
functions to a database of known functions, to aid in the process of reverse engineering.

This mono-repo consists of three different sections, for more information about each of these check their respective READMEs:

- [**asmtransformers**](./asmtransformers) - Training and inference code for the machine learning model
- [**citatio**](./citatio) - A FastAPI backend for the project (this depends on *asmtransformers*)
- [**sententia**](./sententia) - A Ghidra frontend to interface with the service and model mentioned above

The corresponding models, called `ARM64BERT` and `ARM64BERT-embedding` are available on [Hugging Face 🤗](https://huggingface.co/collections/NetherlandsForensicInstitute/arm64bert-6825cca70b6b855fbe4b347b).
