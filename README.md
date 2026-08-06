ASMTransformers 🦾
==================

ASMTransformers is a project to train and use a machine learning model to compare assembly (ARM64, AMD64, i386, RISC-V)
functions to a database of known functions, to aid in the process of reverse engineering.

This mono-repo consists of three different sections, for more information about each of these check their respective READMEs:

- [**asmtransformers**](./asmtransformers) - Training and inference code for the machine learning model
- [**citatio**](./citatio) - A FastAPI backend for the project (this depends on *asmtransformers*)
- [**sententia**](./sententia) - A Ghidra frontend to interface with the service and model mentioned above

The multilingual model, called `Multilingual ASMBERT` is available on [Hugging Face 🤗](https://huggingface.co/NetherlandsForensicInstitute/Multilingual-ASMBERT)

The  older, monolingual models, called `ARM64BERT` and `ARM64BERT-embedding` are also still available on [Hugging Face 🤗](https://huggingface.co/collections/NetherlandsForensicInstitute/arm64bert-6825cca70b6b855fbe4b347b). These models were created using the v1.0.0 release of this repo.

Examples
--------

To see a minimally working example, you need to do two things:

1. Set up the [**citatio**](./citatio) back-end by following the steps described in that folder. The end result is a server running at port 8000.

2. Set up the [**sententia**](./sententia) Ghidra plugin by following the step described in that folder. Configure the plugin so that the port matches the **citatio** server port. 

You can now load up an arm64 binary, start adding functions to the (ephemeral!) database via the **sententia** plugin, and locally request the model for similarity scores with other functions.

A more meaningful example with a prepopulated database is provided in [./examples/atf](), to set this up see the instructions over there.
