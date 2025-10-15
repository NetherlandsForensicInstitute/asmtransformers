ASMTransformers 🦾
==================

ASMTransformers is a project to train and use a machine learning model to compare assembly (currently only ARM64)
functions to a database of known functions, to aid in the process of reverse engineering.

This mono-repo consists of three different sections, for more information about each of these check their respective READMEs:

- [**asmtransformers**](./asmtransformers) - Training and inference code for the machine learning model
- [**citatio**](./citatio) - A FastAPI backend for the project (this depends on *asmtransformers*)
- [**sententia**](./sententia) - A Ghidra frontend to interface with the service and model mentioned above

The corresponding models, called `ARM64BERT` and `ARM64BERT-embedding` are available on [Hugging Face 🤗](https://huggingface.co/collections/NetherlandsForensicInstitute/arm64bert-6825cca70b6b855fbe4b347b).

Examples
--------

To see a minimally working example, you need to do two things:

1. Set up the [**citatio**](./citatio) back-end by following the steps described in that folder. The end result is a server running at port 8000.

2. Set up the [**sententia**](./sententia) Ghidra plugin by following the step described in that folder. Configure the plugin so that the port matches the **citatio** server port. 

You can now load up an arm64 binary, start adding functions to the (ephemeral!) database via the **sententia** plugin, and locally request the model for similarity scores with other functions.

A more meaningful example with a prepopulated database is provided in [./examples/atf](), to set this up see the instructions over there.