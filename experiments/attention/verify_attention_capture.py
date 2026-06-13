"""verify_attention_capture.py — token-attention visualization, done rigorously.

Lessons baked in (from the verification):
  - N_img is COMPUTED from the tensor S (= S - 99), not hardcoded to 768
  - language tokens labeled via openpi's REAL PaligemmaTokenizer (right-pad+add_bos),
    NOT HuggingFace AutoTokenizer (which left-pads and gave us the wrong layout)
  - attention pulled from cap['captures'] (the real nesting)
  - the BOS position is the FIRST language token (right-padding)

Run on REMOTE with venv python:
    ~/flare/openpi/.venv/bin/python ~/flare/visualize_attention_verified.py

Set LAYER_NUM to inspect a different layer.
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAP_PATH = os.path.expanduser("~/flare/results/mechanism_probe_T0_v1/captured_a.pt")
OUT_DIR = os.path.expanduser("~/flare/figures")
os.makedirs(OUT_DIR, exist_ok=True)
LAYER_NUM = "17"          # expert layer to visualize
LANG_LEN, STATE_LEN, ACT_LEN = 48, 1, 50   # from config: max_token_len=48, 1 state, action_horizon=50

# ---------------------------------------------------------------------------
# 1. Load captured attention
# ---------------------------------------------------------------------------
cap = torch.load(CAP_PATH, map_location="cpu")
prompt = cap.get("prompt", "<unknown>")
captures = cap["captures"]
print(f"prompt: {prompt!r}")
print(f"n capture entries: {len(captures)}")

# find the expert layer LAYER_NUM
target = None
for k in captures:
    if "expert" in k.lower() and f".{LAYER_NUM}." in str(k):
        target = k
        break
if target is None:
    print("\nNo match. Expert-related keys available:")
    for k in captures:
        if "expert" in str(k).lower():
            v = captures[k]
            shp = getattr(v, "shape", f"{type(v).__name__}"
                          + (f"[{len(v)}]" if isinstance(v, (list, tuple)) else ""))
            print(f"   {k}: {shp}")
    raise SystemExit("Set LAYER_NUM to one of the above.")

att = captures[target]
print(f"\n[use] {target}")
# handle list-of-calls (per denoise step) or single tensor
if isinstance(att, (list, tuple)):
    print(f"   stored as list of {len(att)} calls; averaging")
    att = torch.stack([torch.as_tensor(a).float() for a in att]).mean(0)
att = torch.as_tensor(att).float()
print(f"   raw shape: {tuple(att.shape)}")
while att.dim() > 3:
    att = att.mean(0)
if att.dim() == 3:        # [H, T, S]
    att = att.mean(0)     # -> [T, S]
T, S = att.shape
print(f"   collapsed: T(queries)={T}  S(keys)={S}")

# ---------------------------------------------------------------------------
# 2. Region boundaries — COMPUTED from S (not hardcoded)
# ---------------------------------------------------------------------------
N_img = S - (LANG_LEN + STATE_LEN + ACT_LEN)
img_end = N_img
lang_end = N_img + LANG_LEN
state_pos = lang_end
act_start = lang_end + STATE_LEN
print(f"   => N_img (image tokens) = S - 99 = {N_img}  ({N_img // 256} images × 256)")
print(f"   image [0:{img_end})  language [{img_end}:{lang_end})  state [{state_pos}]  action [{act_start}:{S})")

# sanity: row-sum should be ~1 if real post-softmax attention
print(f"   row-sum (should be ~1.0): {float(att[-1].sum()):.3f}")

# attention RECEIVED per key, from action queries (last ACT_LEN rows)
if T >= ACT_LEN:
    recv = att[-ACT_LEN:, :].mean(0).numpy()
else:
    recv = att.mean(0).numpy()

# ---------------------------------------------------------------------------
# 3. Language token labels via openpi's REAL tokenizer
# ---------------------------------------------------------------------------
lang_label = {}   # absolute key position -> decoded token
tokenizer_ok = False
for import_path in [
    "openpi.models.tokenizer",
    "openpi.transforms",
    "openpi.shared.tokenizer",
]:
    try:
        mod = __import__(import_path, fromlist=["PaligemmaTokenizer"])
        PaligemmaTokenizer = getattr(mod, "PaligemmaTokenizer")
        tk = PaligemmaTokenizer(LANG_LEN) if "max" not in PaligemmaTokenizer.__init__.__code__.co_varnames[:2] \
             else PaligemmaTokenizer(max_len=LANG_LEN)
        out = tk.tokenize(prompt)
        ids = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
        inner = getattr(tk, "_tokenizer", None)
        for i, tid in enumerate(ids):
            pos = img_end + i
            dec = inner.decode([int(tid)]) if inner is not None else str(int(tid))
            lang_label[pos] = dec
        tokenizer_ok = True
        print(f"   [tokenizer] loaded via {import_path}; {len(ids)} language tokens")
        break
    except Exception as e:
        continue
if not tokenizer_ok:
    print("   [tokenizer] could not import openpi PaligemmaTokenizer — language labels generic. "
          "Edit import_path list to the path you found via gh.")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, (ax, axz) = plt.subplots(2, 1, figsize=(15, 8.5),
                               gridspec_kw={"height_ratios": [2, 1]})
keys = np.arange(S)
ax.axvspan(0, img_end, color="#CDE3F0", alpha=0.5, zorder=0)
ax.axvspan(img_end, lang_end, color="#FFF1C9", alpha=0.65, zorder=0)
ax.axvspan(lang_end, S, color="#D7EFD2", alpha=0.5, zorder=0)
ax.plot(keys, recv, color="#333333", lw=0.7, zorder=2)
ax.fill_between(keys, 0, recv, color="#666666", alpha=0.3, zorder=1)

# top sinks
top = recv.argsort()[::-1][:8]
for p in top:
    ax.plot(p, recv[p], "o", color="#CC3311", markersize=7, zorder=3)
    if p < img_end:
        lbl = f"key {p}\nimage patch"
    elif p < lang_end:
        tok = lang_label.get(p, f"lang+{p - img_end}")
        bos_tag = " (BOS)" if (p == img_end) else ""
        lbl = f"key {p}\n{tok!r}{bos_tag}"
    elif p == state_pos:
        lbl = f"key {p}\nstate"
    else:
        lbl = f"key {p}\naction {p - act_start}"
    ax.annotate(lbl, xy=(p, recv[p]),
                xytext=(p, recv[p] + recv.max() * 0.09),
                fontsize=8, color="#CC3311", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#CC3311", lw=0.8))

ymax = recv.max() * 1.28
ax.text(img_end / 2, ymax * 0.95, f"IMAGE ({N_img} tokens = {N_img//256}×256)",
        ha="center", fontsize=10, fontweight="bold", color="#2C5777")
ax.text((img_end + lang_end) / 2, ymax * 0.95, "LANGUAGE",
        ha="center", fontsize=10, fontweight="bold", color="#8A6D00")
ax.text((lang_end + S) / 2, ymax * 0.95, "STATE+ACTION",
        ha="center", fontsize=10, fontweight="bold", color="#3A6B33")
ax.set_xlim(0, S)
ax.set_ylim(0, ymax)
ax.set_xlabel("Key position (full sequence)", fontsize=11)
ax.set_ylabel("Attention received\n(from action queries)", fontsize=10)
ax.set_title(f"Base pi0_libero — attention per key (expert L{LAYER_NUM})\n"
             f"prompt: {prompt!r}", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# zoom: language region with real token labels
lang_keys = np.arange(img_end, lang_end)
lang_attn = recv[img_end:lang_end]
colors = []
for p in lang_keys:
    tok = lang_label.get(p, "")
    if tok == "<pad>" or tok == "":
        colors.append("#BBBBBB")
    elif p == img_end:
        colors.append("#2E7D32")   # BOS
    else:
        colors.append("#4477AA")   # content
axz.bar(lang_keys, lang_attn, color=colors, edgecolor="black", linewidth=0.3)
for p in lang_keys:
    tok = lang_label.get(p, "")
    if tok and (lang_attn[p - img_end] > lang_attn.max() * 0.1 or p == img_end):
        axz.text(p, lang_attn[p - img_end] + lang_attn.max() * 0.03, tok,
                 rotation=90, fontsize=7, ha="center", va="bottom")
axz.set_xlim(img_end - 0.5, lang_end - 0.5)
axz.set_xlabel(f"Language keys [{img_end}:{lang_end}]  —  green=BOS, blue=content, gray=pad", fontsize=10)
axz.set_ylabel("Attention", fontsize=10)
axz.set_title("Zoom: language region (BOS at first position under openpi right-padding)",
              fontsize=11, fontweight="bold")
axz.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(OUT_DIR, f"attention_tokens_verified_L{LAYER_NUM}.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\n[wrote] {out}")

# numeric dump of the sinks with verified identities
print("\nTop attended keys (verified identities):")
for p in top:
    if p < img_end:
        ident = f"image patch (img {p//256}, patch {p%256})"
    elif p < lang_end:
        ident = f"{lang_label.get(p, '?')!r}" + (" = BOS" if p == img_end else "")
    elif p == state_pos:
        ident = "state token"
    else:
        ident = f"action token {p - act_start}"
    print(f"  key {p:4d}: attn={recv[p]:.4f}  {ident}")
