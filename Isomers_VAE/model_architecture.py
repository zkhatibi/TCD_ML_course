import torch.nn as nn
import torch

# VAE model:
class VAE_model(nn.Module):
    def __init__(self, input_dim, latent_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        hidden_dim = int((latent_dim*2+64)) 
        hidden_dim2 = int((latent_dim+64)) 
        self.mean = nn.Parameter(torch.ones(latent_dim))
        self.sigma = nn.Parameter(torch.ones(latent_dim))
        
        # GRU Encoder
        self.encoder_gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.encoder_gru2 = nn.GRU(2*hidden_dim, hidden_dim2, batch_first=True, bidirectional=True) 
        self.encoder_gru3 = nn.GRU(2*hidden_dim2, latent_dim, batch_first=True, bidirectional=True)
        self.encoder_dense_mu = nn.Linear(2*latent_dim, latent_dim)
        self.encoder_dense_logvar = nn.Linear(2*latent_dim, latent_dim)
        self.encoder_dropout = nn.Dropout(p=0.2)
        
        # GRU Decoder
        self.decoder_gru1 = nn.GRU(latent_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder_gru2 = nn.GRU(2*hidden_dim, input_dim, batch_first=True, bidirectional=True)
        self.decoder_dense = nn.Linear(2*input_dim, input_dim)
        self.decoder_dropout = nn.Dropout(p=0.2)
        
        # Regressor
        self.regressor_linear1 = nn.Linear(latent_dim, latent_dim)
        self.regressor_act1 = nn.LeakyReLU()
        self.regressor_dropout = nn.Dropout(p=0.005)
        self.regressor_linear2 = nn.Linear(latent_dim, 8)
        self.regressor_act2 = nn.SiLU()
        self.regressor_linear3 = nn.Linear(8, 1)
        
    def encode(self, x):
        out, _ = self.encoder_gru1(x) 
        out = self.encoder_dropout(out)
        out, _ = self.encoder_gru2(out)
        out = self.encoder_dropout(out)
        _, h= self.encoder_gru3(out)
        h_cat = torch.cat((h[0,:,:], h[1,:,:]), dim=-1)
        mu = self.encoder_dense_mu(h_cat).squeeze(0)
        logvar = self.encoder_dense_logvar(h_cat).squeeze(0)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = z.unsqueeze(1)  
        h = h.repeat(1, self.seq_len, 1)
        out, _ = self.decoder_gru1(h)
        out = self.decoder_dropout(out)
        recon, _ = self.decoder_gru2(out) 
        recon = self.decoder_dense(recon)
        return recon  
    
    def normalize_latent(self, z):
        return (z-self.mean)/(1e-14 + self.sigma)
    
    def regressor(self, z):
        out = self.regressor_linear1(z)
        out = self.regressor_act1(out)
        out = self.regressor_dropout(out)
        out = self.regressor_linear2(out)
        out = self.regressor_act2(out)
        out = self.regressor_dropout(out)
        out = self.regressor_linear3(out)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        z_norm = self.normalize_latent(z)
        y = self.regressor(z_norm)
        return recon, y, mu, logvar, z_norm, z