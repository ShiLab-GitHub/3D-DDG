def DGSDTAModelConfig():
    config = {}
    config['graphNet'] = 'GAT'
    config['seqNet'] = 'albert'
    config['dropout'] = 0.2
    config['graph_output_dim'] = 128
    config['graph_features_dim'] = 36  # 这个暂时修改为32
    config['seq_embed_dim'] =  4096  # if albert seq_embed_dim = 4096, else 1024
    config['n_filters'] = 256
    config['seq_output_dim'] = 1024
    return config

