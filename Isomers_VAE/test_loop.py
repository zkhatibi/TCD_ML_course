import torch
from rdkit import Chem
import torch.nn as nn
import numpy as np


def test_loop(model, test_loader, idx_to_char, model_checkpoint_path, out_dir):
    # checkpoint = torch.load(model_checkpoint_path)  # Load full checkpoint
    # model.load_state_dict(checkpoint["model_state_dict"])  # Load only model weights

    model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True))

    # Evaluating the testset:
    model.eval()

    # Loop over the test set
    with torch.no_grad():  # Disable gradient calculation for testing
        for batch_data in test_loader:
            x, mask, rbf, sbf, elements, bond_list, sbf_angle_list, _, _, _, atom_numbers = batch_data
            target = torch.argmax(x, dim=2)
            recon, _, _, _, _, _ = model(x, mask, elements, rbf, sbf, bond_list, sbf_angle_list, atom_numbers)
            recon = nn.Softmax(dim=2)(recon)
            recon = torch.argmax(recon, dim=2)
    smiles_org = []
    smiles_recon = []

    for i, mol in enumerate (target):
        smiles_org.append(''.join([idx_to_char[idx.item()] for idx in mol]).replace(' ',''))
        smiles_recon.append(''.join([idx_to_char[idx.item()] for idx in recon[i]]).replace(' ',''))


    valid_smiles = [string for string in smiles_recon if Chem.MolFromSmiles(string)]
    correct_smiles = [string for it, string in enumerate(smiles_recon) if string == smiles_org[it]]

    print(f'Percentage of correctly reconstructed molecules: {100*len(correct_smiles)/len(target)}, Percentage of valid molecules: {100*len(valid_smiles)/len(target)}')

    if valid_smiles:
        print('Some examples of the valid structures:')
        [print(string) for string in valid_smiles[:50]]
    if correct_smiles:
        print('Some examples of the correct structures:')
        [print(string) for string in correct_smiles[:50]]
    
    np.savetxt(out_dir+'stat.txt', [100*len(correct_smiles)/len(target), 100*len(valid_smiles)/len(target)])