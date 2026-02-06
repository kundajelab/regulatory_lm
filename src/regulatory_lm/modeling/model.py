import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rotary_embedding_torch import RotaryEmbedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class RegulatoryLM(torch.nn.Module):
    def __init__(self, input_embedder, encoder, decoder, species_emb_as_mask=False):
        '''
        This is the base RegulatoryLM class. A model contains three components:
        -input_embedder, which converts input tokens to embeddings
        -encoder, which is the bulk of the model
        -decoder, which converts model embeddings to token predictions
        -See the example config file for the arguments used in the paper
        '''
        super().__init__()
        self.species_emb_as_mask = species_emb_as_mask

        self.input_embedder = input_embedder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, seqs, categories):
        '''
        This is a bit convoluted because we initially had a model that returned masked and unmasked embeddings at the same time
        Assume species_emb_as_mask is False and embs_masked is the normal model embeddings from the masked input
        '''
        seq_emb, species_emb = self.input_embedder(seqs, categories)
        if self.species_emb_as_mask:
            embs, embs_masked = self.encoder(seq_emb, mask_vals=species_emb)
        else:
            embs, embs_masked = self.encoder(seq_emb)
        logits_masked = self.decoder(embs_masked)
        return logits_masked
    
    def embed(self, seqs, categories, masked=False):
        input_embs, _ = self.input_embedder(seqs, categories)
        embs, embs_masked = self.encoder(input_embs)
        return embs_masked if masked else embs

class RegulatoryLMEveryLayerEmbedding(torch.nn.Module):
    def __init__(self, input_embedder, encoder, decoder, species_emb_as_mask=False):
        super().__init__()
        self.species_emb_as_mask = species_emb_as_mask

        self.input_embedder = input_embedder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, seqs, categories):
        '''
        This is a bit convoluted because we initially had a model that returned masked and unmasked embeddings at the same time
        Assume species_emb_as_mask is False and embs_masked is the normal model embeddings from the masked input
        '''
        seq_emb, cat_emb = self.input_embedder(seqs, categories)
        if self.species_emb_as_mask:
            embs, embs_masked = self.encoder(seq_emb, cat_emb, mask_vals=species_emb)
        else:
            embs, embs_masked = self.encoder(seq_emb, cat_emb)
        logits_masked = self.decoder(embs_masked)
        return logits_masked
    
    def embed(self, seqs, categories, masked=False):
        input_embs, cat_embs = self.input_embedder(seqs, categories)
        embs, embs_masked = self.encoder(input_embs, cat_embs)
        return embs_masked if masked else embs

class RegulatoryLMWithClassification(torch.nn.Module):
    def __init__(self, input_embedder, encoder, decoder, classifier, species_emb_as_mask=False):
        super().__init__()
        self.species_emb_as_mask = species_emb_as_mask

        self.input_embedder = input_embedder
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def forward(self, seqs, categories):
        seq_emb, species_emb = self.input_embedder(seqs, categories)
        if self.species_emb_as_mask:
            embs, embs_masked = self.encoder(seq_emb, mask_vals=species_emb)
        else:
            embs, embs_masked = self.encoder(seq_emb)
        logits_masked = self.decoder(embs_masked)
        peak_probs = self.classifier(embs)
        return logits_masked, peak_probs
    
    def embed(self, seqs, categories, masked=False):
        input_embs = self.input_embedder(seqs, categories)
        embs, embs_masked = self.encoder(input_embs)
        return embs_masked if masked else embs


class RegulatoryLMClassifierOnly(torch.nn.Module):
    def __init__(self, input_embedder, encoder, classifier, species_emb_as_mask=False):
        super().__init__()
        self.species_emb_as_mask = species_emb_as_mask

        self.input_embedder = input_embedder
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, seqs, categories):
        seq_emb, species_emb = self.input_embedder(seqs, categories)
        if self.species_emb_as_mask:
            embs, embs_masked = self.encoder(seq_emb, mask_vals=species_emb)
        else:
            embs, embs_masked = self.encoder(seq_emb)
        peak_probs = self.classifier(embs)
        return peak_probs
    
    def embed(self, seqs, categories, masked=False):
        input_embs = self.input_embedder(seqs, categories)
        embs, embs_masked = self.encoder(input_embs)
        return embs_masked if masked else embs


class InputEmbedder(torch.nn.Module):
    def __init__(self, emb_size, num_categories, vocab_size=5, masking=False):
        super().__init__()
        if not masking:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-1)
        else:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)
        self.cat_emb = torch.nn.Embedding(num_categories, emb_size)

    def forward(self, seqs, species):
        seq_emb = self.vocab_emb(seqs)
        species_emb = self.cat_emb(species)
        total_emb = seq_emb + species_emb.unsqueeze(1)

        return total_emb, species_emb

class InputEmbedderWithScaledCat(torch.nn.Module):
    def __init__(self, emb_size, num_categories, vocab_size=5, masking=False):
        super().__init__()
        if not masking:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-1)
        else:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)
        self.cat_emb = torch.nn.Embedding(num_categories, emb_size)
        self.cat_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, seqs, species):
        seq_emb = self.vocab_emb(seqs)
        species_emb = self.cat_emb(species)
        total_emb = seq_emb + self.cat_scale * species_emb.unsqueeze(1)

        return total_emb, species_emb


class InputSeqOnlyEmbedder(torch.nn.Module):
    def __init__(self, emb_size, vocab_size=5, masking=False):
        super().__init__()
        if not masking:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-1)
        else:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)

    def forward(self, seqs, species):
        seq_emb = self.vocab_emb(seqs)

        return seq_emb, None

class InputSeqCellTypeEmbedder(torch.nn.Module):
    def __init__(self, emb_size, cell_emb_input_size, project_cell_emb=True, vocab_size=5, masking=False):
        super().__init__()
        if not masking:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-1)
        else:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)
        self.project_cell_emb = project_cell_emb
        if self.project_cell_emb:
            self.proj_layer = torch.nn.Linear(cell_emb_input_size, emb_size)
        else:
            assert cell_emb_input_size == emb_size

    def forward(self, seqs, cell_emb):
        seq_emb  = self.vocab_emb(seqs)
        if self.project_cell_emb:
            cell_emb = self.proj_layer(cell_emb)
        total_emb = seq_emb + cell_emb.unsqueeze(1)

        return total_emb, cell_emb

class InputSeqCellTypeEmbedderWithPE(torch.nn.Module):
    def __init__(self, emb_size, cell_emb_input_size, project_cell_emb=True, vocab_size=5, masking=False, dropout_prob=0.1):
        super().__init__()
        if not masking:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-1)
        else:
            self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)
        self.project_cell_emb = project_cell_emb
        if self.project_cell_emb:
            self.proj_layer = torch.nn.Linear(cell_emb_input_size, emb_size)
        else:
            assert cell_emb_input_size == emb_size

        self.pos_emb = PositionalEncoding(emb_size)
        self.dropout  = torch.nn.Dropout(p=dropout_prob)
        self.layernorm = torch.nn.LayerNorm(emb_size, eps=1e-12)


    def forward(self, seqs, cell_emb):
        seq_emb  = self.vocab_emb(seqs)
        if self.project_cell_emb:
            cell_emb = self.proj_layer(cell_emb)
        total_emb = seq_emb + cell_emb.unsqueeze(1)
        total_emb = self.pos_emb(total_emb)
        total_emb = self.dropout(self.layernorm(total_emb))
        return total_emb, cell_emb


class InputBertEmbedder(torch.nn.Module):
    def __init__(self, emb_size, num_categories, seq_len, vocab_size=6, masking=True):
        super().__init__()
        self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)
        self.cat_emb = torch.nn.Embedding(num_categories, emb_size)
        self.pos_emb = torch.nn.Embedding(seq_len, emb_size)

    def forward(self, seqs, species):
        seq_emb = self.vocab_emb(seqs)
        species_emb = self.cat_emb(species)
        posit_emb = self.pos_emb.weight.repeat(seqs.shape[0], 1, 1)
        total_emb = seq_emb + species_emb.unsqueeze(1) + posit_emb
        return total_emb, species_emb

class InputBertSeqOnlyEmbedder(torch.nn.Module):
    def __init__(self, emb_size, seq_len, vocab_size=6, masking=True, dropout_prob=0.1):
        '''
        Index 4 corresponds to special characters (ie. "N") and are set as padding and ignored
        '''
        super().__init__()
        self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)
        self.pos_emb = torch.nn.Embedding(seq_len, emb_size) #Learnable positional encoding
        self.dropout  = torch.nn.Dropout(p=dropout_prob)
        self.layernorm = torch.nn.LayerNorm(emb_size, eps=1e-12)

    def forward(self, seqs, species):
        seq_emb = self.vocab_emb(seqs)
        posit_emb = self.pos_emb.weight.repeat(seqs.shape[0], 1, 1)
        total_emb = seq_emb + posit_emb
        total_emb = self.dropout(self.layernorm(total_emb))
        return total_emb, None

class InputEmbedderWithPE(torch.nn.Module):
    def __init__(self, emb_size, vocab_size=6, dropout_prob=0.1, masking=True):
        super().__init__()
        self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)
        self.pos_emb = PositionalEncoding(emb_size)
        self.dropout  = torch.nn.Dropout(p=dropout_prob)
        self.layernorm = torch.nn.LayerNorm(emb_size, eps=1e-12)

    def forward(self, seqs, species):
        seq_emb = self.vocab_emb(seqs)
        total_emb = self.pos_emb(seq_emb)
        total_emb = self.dropout(self.layernorm(total_emb))
        return total_emb, None

class InputEmbedderWithCatEmbAndPE(torch.nn.Module):
    def __init__(self, emb_size, num_cats=170, vocab_size=6, dropout_prob=0.1, masking=True):
        super().__init__()
        self.vocab_emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size-2)
        self.cat_emb = torch.nn.Embedding(num_cats, emb_size)
        self.pos_emb = PositionalEncoding(emb_size)
        self.dropout  = torch.nn.Dropout(p=dropout_prob)
        self.layernorm = torch.nn.LayerNorm(emb_size, eps=1e-12)

    def forward(self, seqs, cats):
        seq_emb = self.vocab_emb(seqs)
        cat_emb = self.cat_emb(cats).unsqueeze(1)
        total_emb = seq_emb + cat_emb
        total_emb = self.pos_emb(total_emb)
        total_emb = self.dropout(self.layernorm(total_emb))
        return total_emb, None


class SimpleLMDecoder(torch.nn.Module):
    def __init__(self, emb_size, vocab_size=4):
        super().__init__()
        self.fc_layer = torch.nn.Linear(emb_size, vocab_size)

    def forward(self, embs):
        return self.fc_layer(embs)

class BERTStyleDecoder(torch.nn.Module):
    def __init__(self, emb_size, vocab_size=4):
        super().__init__()
        self.fc1 = torch.nn.Linear(emb_size, emb_size)
        self.fc2 = torch.nn.Linear(emb_size, vocab_size)
        self.gelu = torch.nn.GELU()
        self.layernorm = torch.nn.LayerNorm(emb_size, eps=1e-12)

    def forward(self, embs):
        first_layer_out = self.layernorm(self.gelu(self.fc1(embs)))
        return self.fc2(first_layer_out)

class PeakClassifier(torch.nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.class_layer = torch.nn.Linear(emb_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, embs):
        return self.class_layer(torch.mean(embs,1)).squeeze()
    
class DilatedConvNet(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, n_filters=256, first_kernel_size=21, residual_kernel_size=3, custom_dilations=None):
        super().__init__()
        self.n_residual_convs = n_encoder_layers - 1
        self.iconv = torch.nn.Conv1d(emb_size, n_filters, kernel_size=first_kernel_size, padding="same")
        self.irelu = torch.nn.ReLU()

        if custom_dilations is None:
            self.rconvs = torch.nn.ModuleList([
                torch.nn.Conv1d(n_filters, n_filters, kernel_size=residual_kernel_size, 
                    dilation=2**i, padding="same") for i in range(n_encoder_layers - 1)
            ])
        else:
            assert len(custom_dilations) == self.n_residual_convs
            self.rconvs = torch.nn.ModuleList([
                torch.nn.Conv1d(n_filters, n_filters, kernel_size=residual_kernel_size, 
                    dilation=custom_dilations[i], padding="same") for i in range(n_encoder_layers - 1)
            ])

        self.rrelus = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(n_encoder_layers - 1)
        ])
        
    def forward(self, x):
        x = torch.transpose(x,1,2)
        x = self.irelu(self.iconv(x))
        for i in range(self.n_residual_convs):
            x_conv = self.rrelus[i](self.rconvs[i](x))
            x = torch.add(x, x_conv)
        embs = torch.transpose(x,1,2)
        return embs, embs


class DilatedConvNetWithEmbeddings(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, n_filters=256, first_kernel_size=21, residual_kernel_size=3, custom_dilations=None):
        super().__init__()
        self.n_residual_convs = n_encoder_layers - 1
        self.iconv = torch.nn.Conv1d(emb_size, n_filters, kernel_size=first_kernel_size, padding="same")
        self.irelu = torch.nn.ReLU()

        if custom_dilations is None:
            self.rconvs = torch.nn.ModuleList([
                torch.nn.Conv1d(n_filters, n_filters, kernel_size=residual_kernel_size, 
                    dilation=2**i, padding="same") for i in range(n_encoder_layers - 1)
            ])
        else:
            assert len(custom_dilations) == self.n_residual_convs
            self.rconvs = torch.nn.ModuleList([
                torch.nn.Conv1d(n_filters, n_filters, kernel_size=residual_kernel_size, 
                    dilation=custom_dilations[i], padding="same") for i in range(n_encoder_layers - 1)
            ])

        self.rrelus = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(n_encoder_layers - 1)
        ])
        
    def forward(self, x, embeddings):
        embeddings = torch.transpose(embeddings.unsqueeze(1),1,2)
        x = torch.transpose(x,1,2)
        x = self.irelu(self.iconv(x))
        x = torch.add(x, embeddings)
        for i in range(self.n_residual_convs):
            x_conv = self.rrelus[i](self.rconvs[i](x))
            x = torch.add(x, x_conv)
            x = torch.add(x, embeddings)
        embs = torch.transpose(x,1,2)
        return embs, embs

class LocalGlobalConvNet(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, custom_dilations, n_filters=256, residual_kernel_size=3):
        super().__init__()
        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=residual_kernel_size, 
                dilation=custom_dilations[i], padding="same") for i in range(n_encoder_layers - 1)
        ])

        self.rrelus = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(n_encoder_layers - 1)
        ])

    def forward(self, x, final_layer=None):
        if final_layer is None:
            final_layer = len(self.rconvs)
        x = torch.transpose(x,1,2)
        for i in range(final_layer):
            x_conv = self.rrelus[i](self.rconvs[i](x))
            x = torch.add(x, x_conv)
        embs = torch.transpose(x,1,2)
        return embs, embs

class LocalGlobalDecoder(torch.nn.Module):
    def __init__(self, emb_size, n_layers, vocab_size=4):
        super().__init__()
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(emb_size, vocab_size) for i in range(n_layers)])

    def forward(self, embs, layer):
        return self.fc_layers[layer](embs)


class BicausalLayer(torch.nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        self.w_left = nn.Parameter(torch.empty(emb_size, emb_size))
        self.w_center = nn.Parameter(torch.empty(emb_size, emb_size))
        self.w_right = nn.Parameter(torch.empty(emb_size, emb_size))
        self.bias = nn.Parameter(torch.empty(emb_size))
        
        init_bound = (emb_size * 3)**(-0.5)
        nn.init.uniform_(self.w_left, -init_bound, init_bound)
        nn.init.uniform_(self.w_center, -init_bound, init_bound)
        nn.init.uniform_(self.w_right, -init_bound, init_bound)
        nn.init.uniform_(self.bias, -init_bound, init_bound)

    def forward(self, x):
        out = F.linear(x[:,1:-1,:,:], self.w_center, self.bias)
        out += F.linear(x[:,2:,:1,:], self.w_right)
        out += F.linear(x[:,:-2,:1,:], self.w_left)

        out = F.relu(out) + x[:,1:-1,:,:]

        return out

class TransformerROPEEncoderLayer(torch.nn.Module):
    def __init__(self, emb_size, heads, dim_feedforward, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, theta=10000, rope_size="full", norm_first=True):
        super().__init__()
        '''
        Creates a single transformer layer with ROPE embeddings
        '''
        self.num_heads = heads
        self.head_dim_size = emb_size // heads
        if rope_size == "full":
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim_size, theta=theta)
        elif rope_size == "half":
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim_size // 2, theta=theta)
        self.self_attn = torch.nn.functional.scaled_dot_product_attention
        self.linear1 = torch.nn.Linear(emb_size, dim_feedforward)
        self.dropout_prob = dropout
        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.linear2 = torch.nn.Linear(dim_feedforward, emb_size)
        self.attn_proj = torch.nn.Linear(emb_size, emb_size)
        self.norm1 = torch.nn.LayerNorm(emb_size, eps=layer_norm_eps, bias=True)
        self.norm2 = torch.nn.LayerNorm(emb_size, eps=layer_norm_eps, bias=True)
        self.activation = activation
        self.q_proj = torch.nn.Linear(emb_size, emb_size)
        self.k_proj = torch.nn.Linear(emb_size, emb_size)
        self.v_proj = torch.nn.Linear(emb_size, emb_size)
        self.norm_first = norm_first

    def split_heads(self, x):
        '''
        Assumes x has input shape [batch_size, seq_len, embed_dim]
        Returns tensor of shape [batch_size, num_heads, seq_len, head_size]
        '''
        x_shape = x.shape
        x_reshaped = x.reshape(x_shape[0], x_shape[1], self.num_heads, self.head_dim_size)
        return torch.permute(x_reshaped, dims=[0,2,1,3])  # [batch, num_heads, seq_len, head_dim]
    
    def combine_heads(self, x):
        '''
        Assumes x has input shape [batch_size, num_heads, seq_len, head_size]
        Returns tensor of shape [batch_size, seq_len, embed_dim]
        '''
        x_permuted = torch.permute(x, dims=[0,2,1,3])  # [batch, seq_len, num_heads, head_size]
        x_perm_shape = x_permuted.shape
        return x_permuted.reshape(x_perm_shape[0], x_perm_shape[1], x_perm_shape[2] * x_perm_shape[3])
    
    def apply_attn_block(self, Q,K,V):
        dropout_p = self.dropout_prob if self.training else 0.0
        mha_out = self.self_attn(Q, K, V, dropout_p=dropout_p)
        return self.dropout(self.attn_proj(self.combine_heads(mha_out)))

    def apply_ff_block(self, x):
        first_linear_out = self.dropout(self.activation(self.linear1(x)))
        second_linear_out = self.dropout(self.linear2(first_linear_out))
        return second_linear_out

    def forward(self, x):
        if self.norm_first:
            x_norm = self.norm1(x)
            Q, K, V = self.q_proj(x_norm), self.k_proj(x_norm), self.v_proj(x_norm)
            Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
            Q, K = self.rotary_emb.rotate_queries_or_keys(Q), self.rotary_emb.rotate_queries_or_keys(K)
            x = x + self.apply_attn_block(Q, K, V)
            x = x + self.apply_ff_block(self.norm2(x))
            return x
        else:
            Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
            Q, K = self.rotary_emb.rotate_queries_or_keys(Q), self.rotary_emb.rotate_queries_or_keys(K)
            x = self.norm1(x + self.apply_attn_block(Q, K, V))
            x = self.norm2(x + self.apply_ff_block(x))
            return x


class TransformerLM(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, heads=8, dim_feedforward=2048, norm_first=True):
        super().__init__()
        transformer_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=dim_feedforward, batch_first=True, norm_first=norm_first)
        self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=n_encoder_layers)
    
    def forward(self, x):
        embs = self.transformer(x)
        return embs, embs


class TransformerLMWithROPE(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, heads=8, dim_feedforward=2048, theta=10000, rope_size="full", norm_first=True):
        super().__init__()
        transformer_stack = torch.nn.ModuleList([
            TransformerROPEEncoderLayer(emb_size=emb_size, heads=heads, dim_feedforward=dim_feedforward, theta=theta, rope_size=rope_size, norm_first=norm_first) for layer in range(n_encoder_layers)
        ])
        self.transformer = torch.nn.Sequential(*transformer_stack)
    
    def forward(self, x):
        embs = self.transformer(x)
        return embs, embs



class TransformerLMWithEmbeddings(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, heads=8, dim_feedforward=2048, norm_first=True):
        super().__init__()
        transformer_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=dim_feedforward, batch_first=True, norm_first=norm_first)
        self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=n_encoder_layers)
    
    def forward(self, x, embeddings):
        embeddings = embeddings.unsqueeze(1)
        for i, layer in enumerate(self.transformer.layers):
            embs  = layer(embs)
            embs = torch.add(embs, embeddings)
        return embs, embs

class CNNTransformerROPEHybrid(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, n_conv_layers=4, n_filters=512, heads=8, first_kernel_size=21, residual_kernel_size=3, dim_feedforward=2048, theta=10000, custom_dilations=None, rope_size="full", 
                 norm_first=True):
        super().__init__()
        self.convnet = DilatedConvNet(emb_size, n_conv_layers, n_filters, first_kernel_size, residual_kernel_size, custom_dilations)
        self.transformer = TransformerLMWithROPE(emb_size, n_encoder_layers-n_conv_layers, heads, dim_feedforward, theta=theta, rope_size=rope_size, norm_first=norm_first)

    def forward(self, x):
        embs, _ = self.convnet(x)
        embs, _ = self.transformer(embs)
        return embs, embs

class GPNTransformerROPEHybrid(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, n_conv_layers, heads=8, kernel_size=9, dim_feedforward=2048, theta=10000, custom_dilations=None, rope_size="full", 
                 norm_first=True):
        super().__init__()
        self.convnet = GPNEncoder(emb_size, n_conv_layers, kernel_size, custom_dilations)
        self.transformer = TransformerLMWithROPE(emb_size, n_encoder_layers-n_conv_layers, heads, dim_feedforward, theta=theta, rope_size=rope_size, norm_first=norm_first)

    def forward(self, x):
        embs, _ = self.convnet(x)
        embs, _ = self.transformer(embs)
        return embs, embs


class CNNTransformerROPEHybridOld(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, n_conv_layers=4, n_filters=512, heads=8, first_kernel_size=21, residual_kernel_size=3, dim_feedforward=2048, theta=10000, custom_dilations=None, rope_size="full", 
                 norm_first=True):
        super().__init__()
        self.convnet = DilatedConvNet(emb_size, n_conv_layers, n_filters, first_kernel_size, residual_kernel_size, custom_dilations)
        self.transformer = TransformerLMWithROPE(emb_size, n_encoder_layers-n_conv_layers, heads, dim_feedforward, theta=theta, rope_size=rope_size, norm_first=norm_first)

    def forward(self, x):
        embs, _ = self.convnet(x)
        embs, _ = self.transformer(x)
        return embs, embs



class CNNTransformerHybrid(torch.nn.Module):
    def __init__(self, emb_size, n_encoder_layers, seq_len, n_conv_layers=4, n_filters=512, heads=8, first_kernel_size=21, residual_kernel_size=3, norm_first=True):
        super().__init__()
        n_transformer_layers = n_encoder_layers - n_conv_layers
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, batch_first=True, norm_first=norm_first)
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=n_transformer_layers)
        self.pos_emb = torch.nn.Embedding(seq_len, emb_size)
        self.n_residual_convs = n_conv_layers - 1
        self.iconv = torch.nn.Conv1d(emb_size, n_filters, kernel_size=first_kernel_size, padding="same")
        self.irelu = torch.nn.ReLU()

        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=residual_kernel_size, 
                dilation=2**i, padding="same") for i in range(n_conv_layers - 1)
        ])
        self.rrelus = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(n_conv_layers - 1)
        ])

    def forward(self, x):
        x = torch.transpose(x,1,2)
        x = self.irelu(self.iconv(x))
        for i in range(self.n_residual_convs):
            x_conv = self.rrelus[i](self.rconvs[i](x))
            x = torch.add(x, x_conv)
        embs = torch.transpose(x,1,2)
        embs += self.pos_emb.weight.repeat(embs.shape[0], 1, 1)
        embs = self.transformer(embs)
        return embs, embs
   
# class BicausalLayer(torch.nn.Module):
#     def __init__(self, emb_size):
#         super().__init__()
#         self.emb_size = emb_size

#         self.filter = nn.Parameter(torch.empty(3, emb_size, emb_size).permute(1,2,0))
#         self.bias = nn.Parameter(torch.empty(emb_size))
        
#         init_bound = (emb_size * 3)**(-0.5)
#         nn.init.uniform_(self.filter, -init_bound, init_bound)
#         nn.init.uniform_(self.bias, -init_bound, init_bound)

#     def forward(self, x):
#         full = x[:,:,0,:]
#         masked = x[:,:,1,:]
#         diff = (masked - full)[:,1:-1,:]

#         full_conv = F.conv1d(full.permute(0,2,1), self.filter, self.bias).permute(0,2,1)
#         center = F.linear(diff, self.filter[:,:,1])
#         masked_conv = full_conv + center

#         out = torch.empty((x.shape[0], x.shape[1]-2, 2, self.emb_size), device=x.device, dtype=x.dtype)
#         out[:,:,0,:] = F.relu(full_conv)
#         out[:,:,1,:] = F.relu(masked_conv)
#         out += x[:,1:-1,:,:]

#         return out

        
class BicausalNet(torch.nn.Module):
    def __init__(self, emb_size, num_layers):
        super().__init__()
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([BicausalLayer(emb_size) for _ in range(num_layers)])

    @staticmethod
    def _scramble_and_pad(x):
        out = torch.zeros((x.shape[0], x.shape[1]+2, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)

        evens = x[:,::2,...]
        odds = x[:,1::2,...]
        
        midpoint = evens.shape[1] + 1
        
        out[:,1:midpoint,...] = evens
        out[:,midpoint:-1,...] = odds
        out[:,0,...] = out[:,-2,...]
        out[:,-1,...] = out[:,1,...]

        return out

    def forward(self, embs, mask_vals=None):
        b, l, c = embs.shape

        if mask_vals is None:
            mask_vals = torch.zeros(b, c, device=embs.device, dtype=embs.dtype)

        if l > 2**self.num_layers:
            raise ValueError(f"Input sequence length {l} is too long for {self.num_layers} layers.")
        if c != mask_vals.shape[1]:
            raise ValueError(f"Embedding size {c} does not match mask size {mask_vals.shape[1]}.")
        if c != self.emb_size:
            raise ValueError(f"Embedding size {c} does not match expected size {self.emb_size}.")
        
        x = mask_vals[:,None,None,:].repeat(1, 2*l+1, 2, 1)
        x[:,1:l+1,0,:] = embs
        x[:,-1,0,:] = embs[:,0,:]

        for layer in self.layers:
            x = layer(x)
            x = self._scramble_and_pad(x)

        x = x[:,1:-1,:,:]
        x = self._scramble_and_pad(x)
        x = x[:,1:l+1,:,:]

        out_full = x[:,:,0,:]
        out_masked = x[:,:,1,:]

        return out_full, out_masked
       

class DummyBicausalNet(BicausalNet):
    def __init__(self, emb_size, num_layers):
        super().__init__(emb_size, num_layers)

    def forward(self, embs, mask_vals=None):
        out_full, out_masked = super().forward(embs, mask_vals)
        return out_full, out_full


# def test_bicausal_net():
#     emb_size = 2
#     num_layers = 3
#     net = BicausalNet(emb_size, num_layers)
#     embs = torch.randn(1, 2**num_layers, emb_size)
#     mask_vals = torch.zeros(1, emb_size)
#     out_full, out_masked = net(embs, mask_vals)
#     print(out_full)
#     print(out_masked)


def profile_bicausal_net():
    import torch
    import torch.autograd.profiler as profiler

    emb_size = 256
    num_layers = 10
    net = BicausalNet(emb_size, num_layers)
    embs = torch.randn(64, 2**num_layers, emb_size)
    mask_vals = torch.zeros(64, emb_size)

    net.to("cuda")
    embs = embs.to("cuda")
    mask_vals = mask_vals.to("cuda")

    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        with profiler.record_function("model_inference"):
            out_full, out_masked = net(embs, mask_vals)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

class TransposeLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return x


class GPNConvLayer(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            TransposeLayer(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                padding="same",
                **kwargs,
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x


class GPNEncoder(nn.Module):
    def __init__(self, emb_size, n_encoder_layers, kernel_size, custom_dilations):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                GPNConvLayer(
                    hidden_size=emb_size,
                    kernel_size=kernel_size,
                    dilation=custom_dilations[i],
                )
                for i in range(n_encoder_layers)
            ]
        )

    def forward(self, x):
        x = self.encoder(x)
        return x, x


MODULES = {"InputEmbedder": InputEmbedder, "SimpleLMDecoder": SimpleLMDecoder, "BicausalNet": BicausalNet, "DummyBicausalNet": DummyBicausalNet, "DilatedConvNet": DilatedConvNet, "CNNTransformerHybrid": CNNTransformerHybrid,
"PeakClassifier":PeakClassifier, "InputBertEmbedder":InputBertEmbedder, "TransformerLM":TransformerLM, "InputSeqOnlyEmbedder":InputSeqOnlyEmbedder, "InputBertSeqOnlyEmbedder":InputBertSeqOnlyEmbedder, "InputEmbedderWithPE":InputEmbedderWithPE,
"GPNEncoder":GPNEncoder, "RegulatoryLMClassifierOnly":RegulatoryLMClassifierOnly, "LocalGlobalConvNet":LocalGlobalConvNet, "InputSeqCellTypeEmbedder":InputSeqCellTypeEmbedder, "InputSeqCellTypeEmbedderWithPE":InputSeqCellTypeEmbedderWithPE,
"RegulatoryLMEveryLayerEmbedding":RegulatoryLMEveryLayerEmbedding, "DilatedConvNetWithEmbeddings":DilatedConvNetWithEmbeddings, "TransformerLMWithEmbeddings":TransformerLMWithEmbeddings, "RegulatoryLM":RegulatoryLM, "RegulatoryLMWithClassification":RegulatoryLMWithClassification,
"BERTStyleDecoder":BERTStyleDecoder, "InputEmbedderWithCatEmbAndPE": InputEmbedderWithCatEmbAndPE, "TransformerLMWithROPE": TransformerLMWithROPE, "CNNTransformerROPEHybrid":CNNTransformerROPEHybrid, "CNNTransformerROPEHybridOld":CNNTransformerROPEHybridOld, "GPNTransformerROPEHybrid":GPNTransformerROPEHybrid,
"InputEmbedderWithScaledCat":InputEmbedderWithScaledCat}



if __name__ == "__main__":
    profile_bicausal_net()
#     test_bicausal_net()