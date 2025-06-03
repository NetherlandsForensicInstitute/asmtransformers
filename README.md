ASMTransformers 🦾
==================

ASMTransformers is a project to train and use a machine learning model to compare assembly (currently only ARM64) 
functions to a database of known functions, to aid in the process of reverse engineering. 

This mono-repo consists of three different sections, for more information about each of these check their respective READMEs:

- [**asmtransformers**](./asmtransformers) - Training and inference code for the machine learning model
- [**citatio**](./citatio) - A FastAPI backend for the project (this depends on *asmtransformers*)
- [**sententia**](./sententia) - A ghidra plug-in 
