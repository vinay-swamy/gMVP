import torch 
from torch import nn 

WINDOW_SIZE = 64 * 2 + 1

class EvolEncoder2(nn.Module):
    def __init__(self, num_species, pairwise_type, weighting_schema, input_shape):
        super.__init()
        self.num_species = num_species
        self.pairwise_type = pairwise_type
        self.weighting_schema = weighting_schema
        if self.weighting_schema == 'spe':
            self.W = nn.parameter.Parameter(torch.zeros((1, num_species)))
        elif self.weighting_schema  == 'none':
            self.W = torch.tensor(1.0 / self.num_species).repeat(self.num_species)
        else:
            raise NotImplementedError
        self.input_shape = input_shape
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
            x2 = torch.matmul(x[:, center_pos:center_pos+1)




