def reset_memory():
    globals().clear()
    import gc
    gc.collect()
# reset_memory()

from data_prep import * 
from model_architecture import *
from training_loop import * 
from test_loop import * 
import torch.optim as optim 
import time 
import os 
import shutil


def reset_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  
    os.makedirs(dir_path)        
    print(f"Setting the output directory: {dir_path}")



if __name__ == '__main__':

    start_time = time.time()

    out_dir = './results/'
    dataset_path = './isomer_data.txt'

    reset_directory(out_dir)
    print('The calculation has started ...')

    # Hyperparameters
    smiles, char_to_idx, idx_to_char, nchars, max_len, prop = define_dict_w_prop(dataset_path)
    parced_data = smiles_to_hot(smiles=smiles, seq_len=max_len, unique_chars=nchars, char_to_idx=char_to_idx)
    splitted_data = DatasetSplitter(parced_data, prop, percentile=0.8, batch_size=100)
    print('The dataloader is ready')

    train = True
    num_data, seq_len, input_dim = np.shape(parced_data)
    epochs = 60
    learning_rate = 1e-3
    latent_dim = 32
    KLD_weight = 1e-2

    print(f"The latent space dim is {latent_dim} and the KL weight is: {KLD_weight}")
    
    # Initialize model, optimizer, and data
    model = VAE_model(input_dim, latent_dim, seq_len)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',        # Use 'min' for loss, 'max' for accuracy
        factor=.5,        # Multiplier for LR. new_lr = lr * factor
        patience=0
    )

    if train:
        training_loop_w_prop(model, optimizer, scheduler, epochs, splitted_data, KLD_weight, out_dir)
    else:
        print('Reading the model checkpoint file to retrive the model parameters...')
        model_checkpoint_path = f'/media/zahra/zahraext/storage/DATA_jupyter/mol_proj/results_n_data/VAE+GNN_results/gnn_min2/{VAE_model}_best_state.pt'
        test_loop(model, splitted_data.testloader, idx_to_char, model_checkpoint_path, out_dir)
        gen_latentZ(model, splitted_data.testloader, model_checkpoint_path, out_dir)
        # get_latent_labels(model, train_loader, model_checkpoint_path, out_dir)



end_time = time.time()
hours = int((end_time-start_time)/3600)
minutes = int(((end_time-start_time)%3600)/60)
print(f'Calculation took {hours} hrs and {minutes} mins.')

reset_memory()
