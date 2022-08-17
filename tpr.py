

import torch
import torch.nn as nn
import torch.nn.functional as F

# Definitions of all the seq2seq models and the TPDN
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Defines the tensor product, used in tensor product representations
class SumFlattenedOuterProduct(nn.Module):
    def __init__(self):
        super(SumFlattenedOuterProduct, self).__init__()
           
    def forward(self, input1, input2):
        sum_outer_product = torch.bmm(input1.transpose(1,2), input2)
        flattened_sum_outer_product = sum_outer_product.view(sum_outer_product.size()[0],-1).unsqueeze(0)
        
        return None, flattened_sum_outer_product

# Like SumFlattenedOuterProduct, but also returns the individual filler/role combos
# The downside is that this is slower than SumFlattenedOuterProduct
class OuterProduct(nn.Module):
    def __init__(self):
        super(OuterProduct, self).__init__()

    def forward(self, input1, input2):
        einsum = torch.einsum('blf,blr->blfr', (input1, input2))
        outputs = einsum.view(einsum.shape[0], einsum.shape[1], -1)

        summed_outputs = torch.sum(outputs, dim=1).unsqueeze(0)

        return outputs.transpose(0,1), summed_outputs

# Tensor Product Encoder
class TensorProductEncoder(nn.Module):
    def __init__(self, hidden_size=None, n_fillers=None, n_roles=None, filler_dim=None, role_dim=None, has_linear_layer=False):
        super(TensorProductEncoder, self).__init__()
	
        self.hidden_size = hidden_size

        # Vocab size for the fillers and roles
        self.n_fillers = n_fillers
        self.n_roles = n_roles

        # Embedding size for fillers and roles
        self.filler_dim = filler_dim
        self.role_dim = role_dim

        # Embeddings for fillers and roles
        # The padding_idx means that the padding token, 0, will be embedded
        # as a vector of all zeroes that get no gradient
        # (so 0 should be reserved for the padding token, not used for any
        # actual vocab items)
        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim, padding_idx=0)
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim, padding_idx=0)

        # Create a layer that will bind together the fillers
        # and roles and then aggregate the filler/role pairs.
        # The default is to use the tensor product as the binding
        # operation and elementwise sum as the aggregation operation.
        # These are implemented together with SumFlattenedOuterProduct()
        self.bind_and_aggregate_layer = SumFlattenedOuterProduct()

        # Final linear layer that takes in the tensor product representation
        # and outputs an encoding of size self.hidden_size
        self.has_linear_layer = has_linear_layer
        if self.has_linear_layer:
            self.output_layer = nn.Linear(self.filler_dim * self.role_dim, self.hidden_size)

    # Function for a forward pass through this layer. Takes a list of fillers and 
    # a list of roles and returns a single vector encoding it.
    def forward(self, fillers, roles):

        fillers = torch.LongTensor(fillers).to(device=device)
        roles = torch.LongTensor(roles).to(device=device)
        
		# Embed the fillers and roles
        fillers = self.filler_embedding(fillers)
        roles = self.role_embedding(roles)

        # Bind and aggregate the fillers and roles
        outputs, hidden = self.bind_and_aggregate_layer(fillers, roles)

        # Pass the encoding through the final linear layer
        if self.has_linear_layer:
            hidden = self.output_layer(hidden)

        # hidden is shape (1, batch_size, hidden_size)
        return hidden



# Tensor Product Decoder
class TensorProductDecoder(nn.Module):
    def __init__(self, hidden_size=None, n_fillers=None, n_roles=None, filler_dim=None, role_dim=None, has_linear_layer=False, tpr_enc_to_invert=None):
        super(TensorProductDecoder, self).__init__()

        self.hidden_size = hidden_size

        self.n_fillers = n_fillers
        self.n_roles = n_roles

        self.filler_dim = filler_dim
        self.role_dim = role_dim

        # Embeddings for fillers and roles
        # The padding_idx means that the padding token, 0, will be embedded
        # as a vector of all zeroes that get no gradient
        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim, padding_idx=0)

        self.has_linear_layer = has_linear_layer
        if self.has_linear_layer:
            self.pre_linear = None
            self.input_layer = nn.Linear(self.hidden_size, self.filler_dim*self.role_dim)


        if tpr_enc_to_invert is not None:
            role_emb_matrix = self.role_embedding.weight.detach()
            
            # The "1:" is to ignore the embedding for the padding token
            role_emb_matrix[1:] = torch.pinverse(tpr_enc_to_invert.role_embedding.weight[1:]).transpose(0,1)
            self.role_embedding.load_state_dict({'weight':role_emb_matrix})
            self.filler_embedding.weight = nn.Parameter(tpr_enc_to_invert.filler_embedding.weight.detach())

            if self.has_linear_layer:
                self.pre_linear = nn.Parameter(-1*tpr_enc_to_invert.output_layer.bias)
                self.input_layer.bias = nn.Parameter(torch.zeros_like(self.input_layer.bias))
                self.input_layer.weight = nn.Parameter(torch.pinverse(tpr_enc_to_invert.output_layer.weight))


    def forward(self, roles, hidden):

        if self.has_linear_layer:
            hidden = self.input_layer(hidden)

        # Reshape the hidden state into a matrix
        # to be viewed as a TPR
        hidden = hidden.transpose(0,1).transpose(1,2)
        hidden = hidden.view(-1,self.filler_dim,self.role_dim)

        roles = torch.LongTensor(roles).to(device=device)

        # Get the role unbinding vectors, and multiply them
        # by the TPR to get the guess for the filler embedding
        roles_emb = self.role_embedding(roles)
        filler_guess = torch.bmm(roles_emb, hidden.transpose(1,2))

        # Find the distance between the filler embedding guess
        # and all actual filler vectors, then use those distance
        # as logits that are the input to the NLLLoss
        filler_guess_orig_shape = filler_guess.shape
        filler_guess = filler_guess.reshape(-1, filler_guess_orig_shape[-1]).unsqueeze(1)
        comparison = self.filler_embedding.weight.unsqueeze(0)

        dists = (filler_guess - comparison).pow(2).sum(dim=2).pow(0.5).reshape(filler_guess_orig_shape[0], filler_guess_orig_shape[1], -1)

        dists = -1*dists.transpose(0,1)
        log_probs = F.log_softmax(dists, dim=2)

        topv, topi = log_probs.transpose(0,1).topk(1)
        preds = topi.squeeze(2).tolist()

        return preds



if __name__ == "__main__":
    # Converts filler/role pairs into vector representations
    # There is a vocab of 7 fillers and 5 roles, and their embedding
    # sizes match their vocab size
    # As it stands, the filler and role embeddings are both randomly initialized
    # (and can be trained via gradient descent). However, it would be easy to modify
    # this to have one or both of them be fixed embeddings that are externally specified.
    # E.g., if the fillers are phonemes, the embedding for each phoneme could be
    # a feature vector for it.
    tpe = TensorProductEncoder(n_fillers=7, n_roles=5, filler_dim=7, role_dim=5)

    # Suppose we want to encode a batch with 2 sequences: 2,4,4,3 and 3,6
    # The sequence elements (represented as integer indices) are the fillers. 
    # The second sequence has been padded with zeroes so that both elements 
    # of the batch are the same size
    fillers = [[2,4,4,3], [3,6,0,0]]

    # The roles match one-to-one with the fillers. So, in the first sequence, 
    # the filler 2 has the role 4; the first filler 4 has the role 3; the second
    # filler 4 has the role 2; and the filler 3 has the role 4.
    # In the second sequence, the roles have been padded with zeroes, as above
    # This uses a "right-to-left" role scheme, where the role is the filler's position
    # in the sequence counting from right to left; but many other role schemes are possible.
    roles = [[4,3,2,1], [2,1,0,0]]

    # Producing a tensor product representation (tpr) from the fillers and roles
    tpr = tpe(fillers, roles)
    
    # This will be a tensor of size (1, batch_size, hidden_size).
    # The hidden size is filler_dim*role_dim = 7*5 = 35
    # The batch size is 2, since we encoded 2 sequences
    print(tpr.shape)

    # Seeing what the tpr itself looks like (really, this is a list of TPRs - one
    # TPR for each element in the batch, in this case 2 elements)
    print(tpr)

    # A TensorProductDecoder can be used to recover the input sequences from 
    # a TPR. We do this by inverting a TensorProductEncoder (passed in as a keyword
    # argument)
    tpd = TensorProductDecoder(n_fillers=7, n_roles=5, filler_dim=7, role_dim=5, tpr_enc_to_invert=tpe)

    # The TensorProductDecoder takes in a list of roles as well as a TPR, and then it tells you
    # what filler is in each of those roles within that TPR.
    # So, if we pass in our full list of roles ("roles"), this should recover the exact
    # list of fillers that we started with.
    print(tpd(roles, tpr))





