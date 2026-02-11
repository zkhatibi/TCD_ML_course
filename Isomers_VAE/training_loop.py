import torch
from loss_func import * 
from copy import deepcopy
import numpy as np 
    
def training_loop_w_prop(model, optimizer, scheduler, epochs, splitted_data, KLD_weight, out_dir):
    
    history = []
    print('Training has started...')

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_points = 0
        best_loss = float('inf')
        best_model_state = None
        
        for batch_data in splitted_data.trainloader:
            x, y = batch_data
            batch_size, _, input_dim = x.size()

            recon, y_pred, mu, logvar, _, _ = model(x)
            recon = recon.reshape(-1, input_dim)  # Logits: (batch_size * seq_len, input_dim)
            target = torch.argmax(x, dim=2).reshape(-1)  # return the indices of chars in the string: (batch_size * seq_len)

            loss = vae_loss(recon, target, y, y_pred, logvar, mu, KLD_weight)
            
            train_loss += loss.detach().item() * batch_size
            total_points += batch_size 
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward() # backprop; evaluates the gradient of the loss function wrt the weights 
            optimizer.step() # gradient decend; takes one step towards minimizing the loss and readjust the weights

        avg_loss = train_loss / total_points

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = deepcopy(model.state_dict()) 


        # Validation set loss:
        model.eval()
        val_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for batch_data in splitted_data.trainloader:
                x, y = batch_data
                batch_size, _, _ = x.size()
                recon, y_pred, mu, logvar, _, _ = model(x)
                recon = recon.reshape(-1, input_dim) 
                target = torch.argmax(x, dim=2).reshape(-1)  
                loss = vae_loss(recon, target, y, y_pred, logvar, mu, KLD_weight)
                val_loss += loss.detach().item() * batch_size
                total_val_samples += batch_size 
                avg_val_loss = val_loss / total_val_samples
            
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg val loss: {avg_val_loss:.4f}")
        history.append([avg_loss, avg_val_loss])
        scheduler.step(avg_loss)  # LR adjustment

    np.savetxt(out_dir+'loss.txt', history)
    # with open(out_dir+'loss.txt', 'w') as file: 
    #     for line in history:
    #         # file.write(' '.join(map(str,line)) +'\n')

    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss.item(), 
    #         'scheduler_state_dict': scheduler.state_dict()
    #     }, out_dir+f'/{model_name}_cp.pt')
    #     print(f"Checkpoint saved at epoch {epoch + 1}")

    # # torch.save(model.state_dict(), out_dir+f'/{model_name}_weight_n_biases.pt')
    torch.save(best_model_state, out_dir+f'/best_state.pt')
    print('Training is done!')
