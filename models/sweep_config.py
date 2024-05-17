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
    'optim':{"values": ['adam','nadam']},
    'learning_rate':{"values": [0.001,0.002,0.0001,0.0002, 0.0005]},
    'embedding_size' : {'values' : [64,128,256,512]},
    'hidden_size' : {'values' : [128,256,512]},
    'dropout' : { 'values' : [0,0.1,0.2,0.5]},
    'num_layers' : {'values' : [1]},
    'bidirectional' : {'values' : [True ,False]},
    'teacher_forcing_ratio':{"values":[0.2,0.5,0.7]}
}
}