"""Lab 3 — hand calculation of parameters and FLOPs (no PyTorch needed).

Convention (from the lab sheet / Ch. "Network Architectures"):
  Linear  : FLOPs = 2 * in * out             (per token / per sample)
  Conv2d  : FLOPs = 2 * K * K * C_in * C_out * H_out * W_out
  Attention: QKV = 2 * S * d * 3d, scores = 2 * S^2 * d, A·V = 2 * S^2 * d,
             out-proj = 2 * S * d * d
Biases, ReLU, pooling, softmax, LayerNorm, GELU are ignored (they are O(N), not O(N*M)).
All numbers are for ONE image (batch = 1).
"""


def linear_params(i, o, bias=True):
    return i * o + (o if bias else 0)


def conv_params(k, cin, cout, bias=True):
    return k * k * cin * cout + (cout if bias else 0)


def mlp():
    layers = [
        ("fc1 784->256", linear_params(784, 256), 2 * 784 * 256),
        ("fc2 256->128", linear_params(256, 128), 2 * 256 * 128),
        ("fc3 128->10", linear_params(128, 10), 2 * 128 * 10),
    ]
    return layers


def cnn():
    return [
        ("conv1 1->16 3x3 @28x28", conv_params(3, 1, 16), 2 * 3 * 3 * 1 * 16 * 28 * 28),
        ("conv2 16->32 3x3 @14x14", conv_params(3, 16, 32), 2 * 3 * 3 * 16 * 32 * 14 * 14),
        ("fc 1568->10", linear_params(32 * 7 * 7, 10), 2 * 32 * 7 * 7 * 10),
    ]


def attention(S=16, P=49, d=64, ff=128, n_layers=2, n_cls=10):
    layers = [
        ("patch embed 49->64 (x16 tokens)", linear_params(P, d), 2 * S * P * d),
        ("pos embedding 16x64", S * d, 0),
    ]
    for l in range(n_layers):
        layers += [
            (f"L{l} QKV proj", linear_params(d, 3 * d), 2 * S * d * 3 * d),
            (f"L{l} scores QK^T", 0, 2 * S * S * d),
            (f"L{l} attn x V", 0, 2 * S * S * d),
            (f"L{l} out proj", linear_params(d, d), 2 * S * d * d),
            (f"L{l} FFN 64->128", linear_params(d, ff), 2 * S * d * ff),
            (f"L{l} FFN 128->64", linear_params(ff, d), 2 * S * ff * d),
            (f"L{l} 2x LayerNorm", 2 * 2 * d, 0),
        ]
    layers.append(("head 64->10", linear_params(d, n_cls), 2 * d * n_cls))
    return layers


MODELS = {"MLP": mlp, "CNN": cnn, "Attention": attention}


def totals():
    out = {}
    for name, fn in MODELS.items():
        rows = fn()
        out[name] = {
            "rows": rows,
            "params": sum(r[1] for r in rows),
            "flops": sum(r[2] for r in rows),
        }
    return out


if __name__ == "__main__":
    for name, t in totals().items():
        print(f"\n=== {name} ===")
        print(f"{'layer':38s}{'params':>10s}{'FLOPs':>14s}")
        for n, p, f in t["rows"]:
            print(f"{n:38s}{p:>10,d}{f:>14,d}")
        print(f"{'TOTAL':38s}{t['params']:>10,d}{t['flops']:>14,d}")
