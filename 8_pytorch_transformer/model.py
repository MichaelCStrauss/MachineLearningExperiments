import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(
        self, d_model=768, n_head=12, n_ctx=1024, d_head=64, bias=True, scale=False
    ):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.c_attn = Conv1D(d_model, d_model * 3)
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )
        self.dropout = nn.Dropout(0.1)
        self.c_proj = Conv1D(d_model, d_model)

    def split_heads(self, x):
        "return shape [`batch`, `head`, `sequence`, `features`]"
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _attn(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.scale:
            scores = scores / math.sqrt(v.size(-1))
        if attn_mask is not None:
            scores = scores + attn_mask
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def forward(self, x, attn_mask=None):
        x = self.c_attn(x)  # new `x` shape - `[1,3,2304]`
        q, k, v = x.split(self.d_model, dim=2)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out = self._attn(q, k, v, attn_mask)
        out = self.merge_heads(out)
        out = self.c_proj(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, num_dimensions, num_heads, ffn, dropout=0.2):
        super(DecoderBlock, self).__init__()
        self.num_dimensions = num_dimensions
        self.ln1 = nn.LayerNorm(num_dimensions)
        self.attention = Attention(num_dimensions, num_heads, 256)
        self.ln2 = nn.LayerNorm(num_dimensions)
        self.feedforward = nn.Linear(num_dimensions, num_dimensions)

    # def forward(self, X, attention_mask=None):
    #     X = self.ln1(X)
    #     attn = self.pre_attention(X)
    #     q, k, v = attn.split(self.num_dimensions, dim=2)
    #     attn = self.attention(q, k, v, key_padding_mask=attention_mask)[0]
    #     X = X + attn
    #     X = X + self.feedforward(self.ln2(X))
    #     return X

    def forward(self, X, attention_mask=None):
        X = self.ln1(X)
        X = X + self.attention(X, attention_mask)
        X = X + self.feedforward(self.ln2(X))
        return X


class TransformerModel(nn.Module):
    def __init__(
        self, vocab_size, num_dimensions, num_heads, ffn, num_layers, dropout=0.2
    ):
        super(TransformerModel, self).__init__()

        self.num_dimensions = num_dimensions
        self.pos_encoder = PositionalEncoding(num_dimensions, dropout)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(num_dimensions, num_heads, ffn, dropout)
                for _ in range(num_layers)
            ]
        )

        self.embeddings = nn.Embedding(vocab_size, num_dimensions)

        self.output = nn.Linear(num_dimensions, vocab_size, bias=False)

    def init_weights(self, embeddings):
        self.apply(self._init_weights)
        self.embeddings.load_state_dict({"weight": embeddings})
        self.output.weight = self.embeddings.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, X, attention_mask=None):
        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(X.shape[0], -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        X = self.embeddings(X) * math.sqrt(self.num_dimensions)
        X = self.pos_encoder(X)
        for block in self.blocks:
            X = block(X, attention_mask)
        output = self.output(X)
        return output


def build_model(device, ntokens):
    emsize = 100  # embedding dimension
    nhid = (
        100  # the dimension of the feedforward network model in nn.TransformerEncoder
    )
    nlayers = 12  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 10  # the number of heads in the multiheadattention models
    dropout = 0.0  # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    return model
