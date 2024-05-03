from typing import Literal
import torch
from torch import nn
import einops

class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        self.img_size = img_size

        self.positional_embedding = nn.Parameter(torch.rand(1, (img_size // patch_size)**2, embed_dim))
        self.class_tokens = nn.Parameter(torch.rand(1, 1, embed_dim))

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, image):
        # Проверка размера изображения
        try:
            image = einops.rearrange(image, "b c h w -> b c h w", h = self.img_size, w = self.img_size)
        except Exception:
            print(f"В будущем тут будет поддержка изображений других размерностей, но пока только {self.img_size}x{self.img_size}")
        
        patches = self.patch_embeddings(image)
        patches = einops.rearrange(patches, "b c h w -> b (h w) c")
        patches = patches + self.positional_embedding.data
        b, h, e = patches.shape
        class_tokens = einops.repeat(self.class_tokens.data, "() h e -> b h e", b=b)
        patches = torch.cat((patches, class_tokens), dim=1)


        return patches

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_layer = nn.GELU()):
        super().__init__()

        if out_features is None:
            out_features = in_features
        
        if hidden_features is None:
            hidden_features = in_features

        # Linear Layers
        self.lin1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.lin2 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features
        )

        # Activation(s)
        self.act = act_layer
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):

        x = self.act(self.dropout(self.lin1(x)))
        x = self.act(self.lin2(x))

        return x

class Attention(nn.Module):
    def __init__(self, dim:int, num_heads=8, qkv_bias=False, attn_drop=0., out_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.soft = nn.Softmax(dim=3) # Softmax по строкам матрицы внимания
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x):

        # Attention
        qkv_after_linear = self.qkv(x)
        qkv_after_reshape = einops.rearrange(qkv_after_linear, "b c (v h w) -> v b h c w", v=3, h=self.num_heads)
        q = qkv_after_reshape[0]
        k = qkv_after_reshape[1]
        k = einops.rearrange(k, "b h c w -> b h w c") # Транспонирование
        v = qkv_after_reshape[2]

        atten = self.soft(torch.matmul(q, k) * self.scale)
        atten = self.attn_drop(atten)
        out = torch.matmul(atten, v)
        out = einops.rearrange(out, "b h c w -> b c (h w)", h=self.num_heads)

        # Out projection
        x = self.out(out)
        x = self.out_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, norm_type:Literal["prenorm", "postnorm"], num_heads=8, mlp_ratio=4, qkv_bias=False, drop_rate=0.):
        super().__init__()
        self.norm_type = norm_type

        # Normalization
        self.norm1 = nn.LayerNorm(
            normalized_shape=dim
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=dim
        )

        # Attention
        self.attention = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            out_drop=drop_rate
        )

        # Dropout
        ...
        
        # MLP
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim // mlp_ratio)
        )


    def forward(self, x):
        if self.norm_type == "prenorm":
            x_inner = self.norm1(x)
            # Attention
            x_inner = self.attention(x_inner)
            x = x_inner + x

            x_inner = self.norm2(x)
            # MLP
            x_inner = self.mlp(x_inner)
            x = x_inner + x
        
        if self.norm_type == "postnorm":
            x_inner = self.attention(x)
            x = x_inner + x
            x = self.norm1(x)
            x_inner = self.mlp(x)
            x = x_inner + x
            x =self.norm2(x)

        return x

class Transformer(nn.Module):
    def __init__(self, depth, dim, norm_type:Literal["prenorm", "postnorm"], num_heads=8, mlp_ratio=4, qkv_bias=False, drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, norm_type, num_heads, mlp_ratio, qkv_bias, drop_rate) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0.0, norm_type = "postnorm"):
        super().__init__()
        # Присвоение переменных
        # Path Embeddings, CLS Token, Position Encoding
        self.patch_emb = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        # Transformer Encoder
        self.transformer = Transformer(
            depth=depth,
            dim=embed_dim,
            norm_type=norm_type,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate
        )
        # Classifier
        self.head = MLP(
            in_features=embed_dim,
            out_features=num_classes,
            drop=drop_rate
        )

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.transformer(x)
        x = self.head(x)
        return x
