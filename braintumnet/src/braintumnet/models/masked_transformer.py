import torch, torch.nn as nn, torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)  # B,C,H',W'
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)  # B,N,C
        x = self.norm(x)
        return x, (H,W)

class SoftMaskGenerator(nn.Module):
    def __init__(self, dim, hidden=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_heads), nn.Sigmoid()
        )
    def forward(self, tokens):  # B,N,C
        m = self.mlp(tokens)    # B,N,H
        return m.permute(0,2,1).contiguous()  # B,H,N

class MaskedSelfAttention(nn.Module):
    def __init__(self, dim, n_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Check if PyTorch 2.0+ scaled_dot_product_attention is available (A100 optimized)
        self.use_sdpa = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, softmask):  # x: B,N,C ; softmask: B,H,N
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.n_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]  # B,H,N,D

        # Use PyTorch 2.0+ Flash Attention if available (A100 optimized)
        if self.use_sdpa and softmask.sum() == (B * self.n_heads * N):  # Only if no masking
            # Standard SDPA (Flash Attention 2 backend on A100)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )
            out = out.transpose(1,2).reshape(B,N,C)
        else:
            # Manual attention with soft masking
            attn = (q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)  # B,H,N,N
            key_bias = torch.log(softmask.unsqueeze(-2) + 1e-6)  # B,H,1,N
            attn = attn + key_bias
            attn = attn.softmax(-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1,2).reshape(B,N,C)

        out = self.proj_drop(self.proj(out))
        return out

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim*mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim*mlp_ratio), dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class MaskedTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MaskedSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
    def forward(self, x, softmask):
        x = x + self.attn(self.norm1(x), softmask)
        x = x + self.mlp(self.norm2(x))
        return x

class AdaptiveMaskedTransformer(nn.Module):
    def __init__(self, in_ch, dim, patch_size=8, depth=2, n_heads=4):
        super().__init__()
        self.pe = PatchEmbed(in_ch, dim, patch_size)
        self.mask_gen = SoftMaskGenerator(dim, hidden=dim//2, n_heads=n_heads)
        self.blocks = nn.ModuleList([MaskedTransformerBlock(dim, n_heads) for _ in range(depth)])
    def forward(self, x):
        tokens, (H,W) = self.pe(x)  # B,N,C
        softmask = self.mask_gen(tokens)  # B,H,N
        for blk in self.blocks:
            tokens = blk(tokens, softmask)
        feat = tokens.transpose(1,2).reshape(x.size(0), tokens.size(-1), H, W)
        return feat
