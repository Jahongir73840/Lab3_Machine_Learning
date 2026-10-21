import argparse
import json
import os
import platform
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import hand_costs

SEED = 0
BATCH = 128


# ----------------------------------------------------------------------------- 1. Data
def get_loaders(root="./data"):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full_train = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    train, val = random_split(full_train, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED))
    kw = dict(batch_size=BATCH, num_workers=2)
    return (DataLoader(train, shuffle=True, **kw), DataLoader(val, **kw), DataLoader(test, **kw))


# ----------------------------------------------------------------------------- 2. Models
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    """28x28 -> 16 patches of 7x7 -> Linear(49,64) + learned pos-emb -> 2x TransformerEncoderLayer -> mean pool -> Linear(64,10).

    A learned positional embedding (16x64 = 1,024 params) is added: without it, self-attention + mean pooling
    is permutation-invariant and the model cannot tell where a patch came from.
    """

    def __init__(self, patch=7, d_model=64, nhead=4, dim_ff=128, layers=2):
        super().__init__()
        self.patch = patch
        n_tokens = (28 // patch) ** 2
        self.embed = nn.Linear(patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):                                   # x: [B,1,28,28]
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)               # [B,1,4,4,7,7]
        x = x.reshape(x.size(0), -1, p * p)                 # [B,16,49]
        x = self.embed(x) + self.pos                        # [B,16,64]
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchAttention}


# ----------------------------------------------------------------------------- 3. Costs
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def torchinfo_summary(model, name, outdir):
    from torchinfo import summary
    s = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                col_names=("input_size", "output_size", "num_params", "mult_adds"))
    with open(os.path.join(outdir, f"summary_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(str(s))
    return s.total_params, s.total_mult_adds


def thop_flops(model, name, outdir):
    """thop reports MACs (multiply-accumulates). FLOPs ~= 2 * MACs."""
    from thop import profile
    import copy
    m = copy.deepcopy(model).eval()
    x = torch.randn(1, 1, 28, 28)
    try:
        macs, params, info = profile(m, inputs=(x,), verbose=False, ret_layer_info=True)
        with open(os.path.join(outdir, f"thop_layers_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(_fmt_thop_info(info))
    except TypeError:  # older thop without ret_layer_info
        macs, params = profile(m, inputs=(x,), verbose=False)
    return float(macs), float(params)


def _fmt_thop_info(info, indent=0):
    lines = []
    for k, v in info.items():
        macs, params, children = v
        lines.append(f"{'  ' * indent}{k:30s} MACs={macs:>12,.0f} params={params:>10,.0f}")
        if children:
            lines.append(_fmt_thop_info(children, indent + 1))
    return "\n".join(lines)


def flopcounter_flops(model):
    """PyTorch's own aten-level counter (counts matmuls incl. attention). Grad is enabled on purpose:
    under no_grad nn.TransformerEncoderLayer takes a fused 'fast path' that the counter cannot see."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return float(fc.get_total_flops())


@torch.inference_mode()
def latency_ms(model, batch, runs, warmup=10):
    model.eval()
    x = torch.randn(batch, 1, 28, 28)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


# ----------------------------------------------------------------------------- 4. Training
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def train(model, train_dl, val_dl, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.numel()
            n += y.numel()
        val_acc = evaluate(model, val_dl)
        history.append({"epoch": ep, "train_loss": loss_sum / n, "val_acc": val_acc, "sec": time.time() - t0})
        print(f"  epoch {ep}: loss={loss_sum / n:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.1f}s)")
    return history


# ----------------------------------------------------------------------------- 5. Plots
def make_plots(res, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res)
    colors = {"MLP": "#2a6fdb", "CNN": "#e07a1f", "Attention": "#2e9e5b"}
    acc = [res[n]["test_acc"] * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("params", "Parameters", lambda n: res[n]["params"]),
        ("lat1", "Latency, batch 1 (ms)", lambda n: res[n]["latency_ms_b1"]),
        ("lat128", "Latency per image, batch 128 (ms)", lambda n: res[n]["latency_ms_b128"] / 128),
    ]
    for ax, (_, xlabel, fx) in zip(axes, panels):
        for n, a in zip(names, acc):
            ax.scatter(fx(n), a, s=120, color=colors[n], zorder=3)
            ax.annotate(n, (fx(n), a), textcoords="offset points", xytext=(8, 6))
        ax.set_xscale("log")
        ax.set_xlabel(xlabel + " (log scale)")
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_title("Accuracy vs parameters")
    axes[1].set_title("Accuracy vs latency (batch 1)")
    axes[2].set_title("Accuracy vs latency (batch 128)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "tradeoffs.png"), dpi=150)
    plt.close(fig)

    # hand vs measured FLOPs
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.25
    xs = range(len(names))
    series = [
        ("Hand calc", [res[n]["flops_hand"] for n in names]),
        ("thop (2 x MACs)", [2 * res[n]["thop_macs"] for n in names]),
        ("torch FlopCounter", [res[n]["flops_flopcounter"] or 0 for n in names]),
    ]
    for i, (label, vals) in enumerate(series):
        ax.bar([x + (i - 1) * w for x in xs], [v / 1e6 for v in vals], w, label=label)
    ax.set_xticks(list(xs), names)
    ax.set_ylabel("MFLOPs per image")
    ax.set_title("Forward FLOPs: hand calculation vs profilers")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "flops_hand_vs_measured.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- 6. Report
def render_report(res, env, template="report_template.md", out="report.md"):
    if not os.path.exists(template):
        print("report_template.md not found, skipping report")
        return
    text = open(template, encoding="utf-8").read()
    vals = {}
    for n, r in res.items():
        k = n.upper()
        vals[f"{k}_PARAMS"] = f"{r['params']:,}"
        vals[f"{k}_PARAMS_TI"] = f"{r['params_torchinfo']:,}"
        vals[f"{k}_FLOPS_HAND"] = f"{r['flops_hand']:,}"
        vals[f"{k}_THOP_MACS"] = f"{r['thop_macs']:,.0f}"
        vals[f"{k}_THOP_FLOPS"] = f"{2 * r['thop_macs']:,.0f}"
        fc = r["flops_flopcounter"]
        vals[f"{k}_FC_FLOPS"] = f"{fc:,.0f}" if fc is not None else "n/a"
        vals[f"{k}_THOP_RATIO"] = f"{2 * r['thop_macs'] / r['flops_hand']:.2f}"
        vals[f"{k}_LAT1"] = f"{r['latency_ms_b1']:.3f}"
        vals[f"{k}_LAT128"] = f"{r['latency_ms_b128']:.2f}"
        vals[f"{k}_LAT128_IMG"] = f"{r['latency_ms_b128'] / 128:.4f}"
        vals[f"{k}_ACC"] = f"{100 * r['test_acc']:.2f}"
        vals[f"{k}_VAL"] = f"{100 * r['history'][-1]['val_acc']:.2f}" if r["history"] else "n/a"
        vals[f"{k}_ACC_PER_100K"] = f"{100 * r['test_acc'] / (r['params'] / 1e5):.1f}"
        vals[f"{k}_GFLOPS_B1"] = f"{r['flops_hand'] / (r['latency_ms_b1'] * 1e-3) / 1e9:.1f}"
        vals[f"{k}_GFLOPS_B128"] = f"{128 * r['flops_hand'] / (r['latency_ms_b128'] * 1e-3) / 1e9:.1f}"
        vals[f"{k}_BATCH_SPEEDUP"] = f"{r['latency_ms_b1'] / (r['latency_ms_b128'] / 128):.0f}"
    best_acc = max(res, key=lambda n: res[n]["test_acc"])
    fastest1 = min(res, key=lambda n: res[n]["latency_ms_b1"])
    fastest128 = min(res, key=lambda n: res[n]["latency_ms_b128"])
    vals["BEST_ACC"] = best_acc
    vals["FASTEST_B1"] = fastest1
    vals["FASTEST_B128"] = fastest128
    vals["ENV"] = env
    for k, v in vals.items():
        text = text.replace("<<" + k + ">>", str(v))
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None, help="fix torch CPU threads for repeatable latency")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild plots and report.md from results/results.json without retraining")
    args = ap.parse_args()

    if args.report_only:
        with open("results/results.json", encoding="utf-8") as f:
            saved = json.load(f)
        os.makedirs("figures", exist_ok=True)
        make_plots(saved["models"], "figures")
        render_report(saved["models"], saved["env"])
        return

    torch.manual_seed(SEED)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    env = (f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}, "
           f"Python {platform.python_version()}, torch {torch.__version__}, CPU threads={torch.get_num_threads()}")
    print(env)

    hand = hand_costs.totals()
    train_dl, val_dl, test_dl = get_loaders()
    res = {}

    for name, cls in MODELS.items():
        print(f"\n===== {name} =====")
        torch.manual_seed(SEED)
        model = cls()
        r = {"params": count_params(model)}
        r["params_torchinfo"], r["torchinfo_mult_adds"] = torchinfo_summary(model, name, "results")
        r["params_hand"] = hand[name]["params"]
        r["flops_hand"] = hand[name]["flops"]
        r["thop_macs"], r["thop_params"] = thop_flops(model, name, "results")
        r["flops_flopcounter"] = flopcounter_flops(model)
        r["latency_ms_b1"] = latency_ms(model, 1, runs=100)
        r["latency_ms_b128"] = latency_ms(model, 128, runs=20)
        print(f"params={r['params']:,} (hand {r['params_hand']:,})  FLOPs hand={r['flops_hand']:,} "
              f"thop 2xMACs={2 * r['thop_macs']:,.0f}  FlopCounter={r['flops_flopcounter']}")
        print(f"latency: b1={r['latency_ms_b1']:.3f} ms, b128={r['latency_ms_b128']:.2f} ms")

        r["history"] = train(model, train_dl, val_dl, args.epochs) if args.epochs > 0 else []
        r["test_acc"] = evaluate(model, test_dl)
        print(f"test accuracy = {r['test_acc']:.4f}")
        res[name] = r

    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump({"env": env, "models": res}, f, indent=2)
    make_plots(res, "figures")
    render_report(res, env)


if __name__ == "__main__":
    main()
