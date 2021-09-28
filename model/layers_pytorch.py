#%%
import torch 
from torch import nn 

WINDOW_SIZE = 64 * 2 + 1

class EvolEncoder2(nn.Module):
    def __init__(self, num_species, pairwise_type, weighting_schema):
        super(EvolEncoder2, self).__init__()
        self.num_species = num_species
        self.pairwise_type = pairwise_type
        self.weighting_schema = weighting_schema
        if self.weighting_schema == 'spe':
            self.W = nn.parameter.Parameter(torch.zeros((1, num_species)))
        elif self.weighting_schema  == 'none':
            self.W = torch.tensor(1.0 / self.num_species).repeat(self.num_species)
        else:
            raise NotImplementedError
       
    def forward(self, x ):
        shape  = x.shape
        B, L, N = shape[0], shape[1], shape[2] // 2
        center_pos = WINDOW_SIZE // 2
        A = 21 + 1  #alphabet size
        A = 21

        ww = x[:, :, 200:]
        w = x[:, 0, 200:]
        x = x[:, :, :200]
        if self.weighting_schema == 'spe':
            #W = self.B * w / 100.0 + self.W  #+ tf.cast(tf.less(w, 0.01),
            #          tf.float32) * -1e12
            W = torch.nn.softmax(self.W)
        else:
            W = self.W
        #x= nn.functional.one_hot(x.int()) #// is this totally necessary? data should be one-hottted before this 
        x1 = torch.matmul(W[:,None, None], x)
        if self.pairwise_type  == 'fre':
            x2 = torch.matmul(x[:, center_pos:center_pos + 1, :, :, None],
                           x[:, :, :, None])
            x2 = torch.reshape(x2, (B, L, N, A * A))
            x2 = torch.matmul(W, x2)
            x2 = torch.squeeze(x2, dim=2)
        elif self.pairwise_type == 'cov':
            #numerical stability
            #(batch_size, window_size, 192, A)
            #VSNOTE: see 
            x2 = x - x1
            x2 = torch.sqrt(W[:, None, :, None]) * x2
            #VSNOTE: I am not totally sure if I got the transposes
            # right here, so if youre looking for an error its probably here 
            x2 = x2.permute((0, 2, 1, 3))
            x2_t = torch.reshape(x2, shape=(B, N, L * A))
            #left(batch_size, 192, A)
            #right(batch_size, 192, L * A)
            #result(batch-size, A, L * A)
            x2 = torch.matmul(torch. x2[:, :, center_pos], x2_t)
            x2 = torch.reshape(x2, (B, A, L, A))
            x2 = x2.permute((0, 2, 1, 3))
            x2 = torch.reshape(x2, (B, L, A * A))
            norm = torch.sqrt(
                torch.sum(torch.square(x2), dim=-1, keepdim=True) + 1e-12)
            x2 = torch.concat([x2, norm], axis=-1)
        elif self.pairwise_type == 'cov_all':
            print('cov_all not implemented in EvolEncoder2')
            raise NotImplementedError
            # #(batch, len, species, 21)
            # x2 = x - x1
            # #(batch, species, len, 21)
            # x2 = tf.transpose(x2, perm=(0, 2, 1, 3))
            # #(batch, species, len * 21)
            # x2 = tf.reshape(x2, shape=(B, N, L * A))
            # x2 = tf.sqrt(W[:, :, tf.newaxis]) * x2
            # x2 = tf.matmul(x2, x2, transpose_a=True)
            # x2 = tf.reshape(x2, (B, L, A, L, A))
            # x2 = tf.transpose(x2, perm=(0, 1, 3, 2, 4))
            # x2 = tf.reshape(x2, (B, L, L, A * A))
            # norm = tf.sqrt(
            #     tf.reduce_sum(tf.square(x2), keepdims=True, axis=-1) + 1e-12)
            # x2 = tf.concat([x2, norm], axis=-1)
        elif self.pairwise_type == 'inv_cov':
            print('in_cov not implemented in EvolEncoder2')
            raise NotImplementedError
            # x2 = x - x1
            # x2 = tf.transpose(x2, perm=(0, 2, 1, 3))
            # x2 = tf.reshape(x2, shape=(B, N, L * A))
            # x2 = tf.sqrt(W[:, :, tf.newaxis]) * x2
            # x2 = tf.matmul(x2, x2, transpose_a=True)
            # x2 += tf.eye(L * A) * 0.01
            # x2 = tf.linalg.inv(x2)
            # x2 = tf.reshape(x2, (B, L, A, L, A))
            # x2 = tf.transpose(x2, perm=(0, 1, 3, 2, 4))
            # x2 = x2[:, center_pos]
            # x2 = tf.reshape(x2, (B, L, A * A))

        elif self.pairwise_type == 'none':
            x2 = None
        else:
            raise NotImplementedError(
                f'pairwise_type {self.pairwise_type} not implemented')

        x1 = torch.squeeze(x1, dim=2)

        return x1, x2

#%%         


l = EvolEncoder2(num_species=200, pairwise_type='cov', weighting_schema='spe')




# %%
