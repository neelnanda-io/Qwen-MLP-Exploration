# %%
import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

# # %%
# x = torch.linspace(-5, 5, 500)
# ys = [F.gelu(x), F.silu(x), F.relu(x)]
# line(x=x, y=ys, title="GELU", line_labels=["Gelu", "Silu", "Relu"])
# line(x=x, y=x * F.gelu(x))
# # %%
# cosines = (
#     (model.W_gate * model.W_in).sum(1)
#     / model.W_gate.norm(dim=1)
#     / model.W_in.norm(dim=1)
# )
# line(cosines, title="Cosine Similarity between W_gate and W_in")
# # %%
# histogram(cosines.T, barmode="overlay", marginal="box", height=1000)

# # %%
# px.box(to_numpy(cosines).T)
# %%
# W_gate_rand = torch.randn_like(model.W_gate)
# W_in_rand = torch.randn_like(model.W_in)
# rand_cosines = (
#     (W_gate_rand * W_in_rand).sum(1) / W_gate_rand.norm(dim=1) / W_in_rand.norm(dim=1)
# )
# px.box(to_numpy(torch.cat([cosines, rand_cosines], dim=0)).T)
# # %%
# logits, cache = model.run_with_cache("Hello World")
# # %%
# for k, v in cache.cache_dict.items():
#     if "blocks.0.mlp" in k:
#         print(k, v.shape)
# # %%
# pre = cache["blocks.0.mlp.hook_pre"]
# pre_linear = cache["blocks.0.mlp.hook_pre_linear"]
# post = cache["blocks.0.mlp.hook_post"]
# %%
dfs = []
for layer in tqdm.trange(n_layers):
    W_gate = model.W_gate[layer]
    W_in = model.W_in[layer]
    W_gate_norm = W_gate / W_gate.norm(dim=0, keepdim=True)
    gate_cosines = W_gate_norm.T @ W_gate_norm
    gate_cosines[torch.arange(d_mlp), torch.arange(d_mlp)] = 0.0
    gate_values, gate_indices = gate_cosines.max(dim=0)
    W_in_norm = W_in / W_in.norm(dim=0, keepdim=True)
    in_cosines = W_in_norm.T @ W_in_norm
    in_cosines[torch.arange(d_mlp), torch.arange(d_mlp)] = 0.0
    in_values, in_indices = in_cosines.max(dim=0)
    cos_in = (W_in_norm * W_in_norm[:, gate_indices]).sum(dim=0)
    temp_df = pd.DataFrame(
        {
            "pair": to_numpy(gate_indices),
            "max_cos_gate": to_numpy(gate_values),
            "cos_in": to_numpy(cos_in),
            "max_cos_in": to_numpy(in_values),
        }
    )
    temp_df["layer"] = layer
    dfs.append(temp_df)
df = pd.concat(dfs)
# %%

# %%
# W_gate_rand = torch.randn_like(model.W_gate[layer])
# W_gate_rand_norm = W_gate_rand / W_gate_rand.norm(dim=0, keepdim=True)
# rand_cosines = W_gate_rand_norm.T @ W_gate_rand_norm
# rand_cosines[torch.arange(d_mlp), torch.arange(d_mlp)] = 0.0
# rand_max_cos, _ = rand_cosines.max(dim=0)
# histogram(rand_max_cos)
# %%
px.box(df, x="layer", y="max_cos_gate", title="max_cos_gate").show()
px.box(df, x="layer", y="max_cos_in", title="max_cos_in").show()
# %%
df[df.layer == 15].pair.value_counts()
# %%
# hist_entries = []
# labels = []
# # %%
# layer = 2
# W_gate = torch.randn_like(model.W_gate[layer])
# # W_gate = torch.cat([W_gate, W_gate], dim=-1)

# W_gate_norm = W_gate / W_gate.norm(dim=0, keepdim=True)
# gate_cosines = W_gate_norm.T @ W_gate_norm
# gate_cosines[torch.arange(d_mlp), torch.arange(d_mlp)] = 0.0
# # histogram(gate_cosines.flatten()[::1001], title="Doubled W_gate")
# hist_entries.append(gate_cosines.flatten()[::1001])
# labels.append("Randn_like")
# # %%
# histogram(torch.stack(hist_entries).T, color=labels)
# %%
layer = 1
W_gate = model.W_gate[layer]
W_gate_cent = W_gate - W_gate.mean(dim=1, keepdim=True)

W_gate_norm = W_gate / W_gate.norm(dim=0, keepdim=True)
gate_cosines = W_gate_norm.T @ W_gate_norm
gate_cosines[torch.arange(d_mlp), torch.arange(d_mlp)] = 0.0
W_gate_cent_norm = W_gate_cent / W_gate_cent.norm(dim=0, keepdim=True)
gate_cent_cosines = W_gate_cent_norm.T @ W_gate_cent_norm
gate_cent_cosines[torch.arange(d_mlp), torch.arange(d_mlp)] = 0.0
histogram(
    torch.stack([gate_cosines.flatten()[::10001], gate_cent_cosines.flatten()[::10001]]).T,
    barmode="overlay",
    title="Layer 1 mean centered W_gate",
)
# hist_entries.append(gate_cosines.flatten()[::1001])
# labels.append("Randn_like")

# %%
Ss = []
for layer in range(n_layers):
    W_gate = model.W_gate[layer]
    U, S, Vh = torch.linalg.svd(W_gate)
    Ss.append(S)
Ss = torch.stack(Ss)
S_norm = Ss / Ss.norm(dim=-1, keepdim=True)
line(S_norm)
line(Ss)
# %%
