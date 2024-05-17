# DeepLearning Assignment 3
Wandb Report Link \
https://wandb.ai/srija17199/DL-Assignment3/reports/Assignment-3--Vmlldzo3OTg3NDk2?accessToken=9zelz3x3j44oaqtm1ukubf3ig03kric61flemnmjigt1n2a04diep8b0ktckpa4x

You can either create a new conda environment and install requirements.txt, or you can use any environment with the libraries specified in requirements.txt

Note that the dataloader, accuracy calculators and Vocab calculations are defined in utils.py and model.py, model_attention.py has the implementation for Encoder, Decoder and EncoderDecoder classes.

## Answer 1
To implement a Vanilla Seq2Seq model that performs transliteration for English to Hindi, run train.py. It takes the arguments given below, along with one default setting.

```
python ./models/train.py
```

```python
    parser = argparse.ArgumentParser()
    parser.add_argument('-ct', '--cell_type', help="Choices:[RNN, LSTM, GRU]", type=str, default='GRU')
    parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=128)
    parser.add_argument('-o', '--optimizer', help = 'choices: [ "adam", "nadam"]', type=str, default = 'adam')
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=2e-4)
    parser.add_argument('-em', '--embedding_size', help='size of embedding', type=int, default=512)
    parser.add_argument('-hd', '--hidden_dim', help='choices:[64,128,256,512]',type=int, default=512)
    parser.add_argument('-dp', '--dropout', help='choices:[0, 0.1, 0.2, 0.3]',type=float, default=0.2)
    parser.add_argument('-nl', '--num_layers', help='Number of layers in network: [1,3,5]',type=int, default=3)
    parser.add_argument('-bidir', '--bidirectional', help='Choices:["True","False"]',type=bool, default=False)
    parser.add_argument('-tf', '--teacher_forcing_ratio', help='choices:[0,0.2,0.3,0.5,0.7]',type=float, default=0.7)
    parser.add_argument('-epc', '--epochs', help = 'Give the number of epochs - ideal between 20 to 30', default = 30)
    parser.add_argument('-pn' , '--project_name', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='CS6910_Assignment3_Q1')
    parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='srija17199')
```

## Answer 2
To run the sweeps for Vanilla Seq2Seq, you can start run_sweep.py. Please provide suitable entity and project name for wandb and relogin to wandb before running the sweeps.

```
python ./models/run_sweep.py
```

```python
sweep_config= {
"name" : "DeepLearning_Assignment3",
"method" : "bayes",
'metric': {
    'name': 'val_acc',
    'goal': 'maximize'
},
'parameters' : {
    'cell_type' : { 'values' : ['RNN','LSTM','GRU'] },
    'batch_size' : {'values' : [32,64,128]},
    'optimizer':{"values": ['adam','nadam']},
    'learning_rate':{"values": [0.005, 0.0002, 0.0005]},
    'embedding_size' : {'values' : [64,128,256,512]},
    'hidden_dim' : {'values' : [128,256,512]},
    'dropout' : { 'values' : [0,0.1,0.2,0.5]},
    'num_layers' : {'values' : [1,3]},
    'bidirectional' : {'values' : [True ,False]},
    'teacher_forcing_ratio':{"values":[0.2, 0.7]},
    }
}
```

### Answer 5

Finally for the attention-based Seq2Seq models, run train_attention.py
```
python ./models/train_attention.py
```

To run the sweep for attention based models, run run_sweep_attention.py

```
python ./models/run_sweep_attention.py
```
