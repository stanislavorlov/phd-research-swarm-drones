# Running pi_tgat_mappo on a Vast.ai GPU instance

I can't reach your Vast.ai instance directly -- I only have a bridge to
files and a shell *on your Mac* (routed through a sandboxed VM with no
outbound SSH/raw-TCP access, confirmed by testing), not to arbitrary
external hosts. So this is a "you run it" deployment: the two scripts in
this folder do the actual work, and you run them from your own Mac
terminal (outside of Claude), where your normal internet/SSH access
already exists.

## What I need from you (none of it is a secret)

1. **The SSH connect string Vast.ai gives you** for the running instance --
   click the "SSH" icon on your instance card at
   https://cloud.vast.ai/instances/. It looks like:
   `ssh -p 50983 root@154.64.230.50`
   I only need the host (`154.64.230.50`) and port (`50983`) from that --
   never a private key or password. Vast.ai auth is via the SSH public key
   you already registered at https://cloud.vast.ai/manage-keys/, which your
   Mac's own `ssh` client presents automatically; nothing needs to be typed
   into or pasted to me.
2. **Which Docker template/image you picked** when creating the instance.
   If it already has PyTorch + CUDA preinstalled (e.g. a `vastai/pytorch:*`
   or `pytorch/pytorch:*-cuda*` template from
   https://cloud.vast.ai/templates/), deployment is faster because `torch`
   is already there. If you picked a bare Ubuntu+CUDA image, that's fine
   too -- the deploy script installs everything needed either way.
3. Confirm the instance has internet egress (default, unless you disabled
   it) so `pip install` can reach PyPI from the instance itself.

## Steps (run these yourself, in your Mac's Terminal.app)

```bash
cd ~/PycharmProjects/phd-research-swarm-drones/kinematic-fiedler/pi_tgat_mappo/deploy

# 1. One-time: sync code, install torch/torch_geometric on the remote GPU
#    box, and run --smoke-test there to confirm the whole pipeline works
#    on CUDA.
./deploy_vastai.sh <HOST> <PORT>

# 2. Launch a real training run that survives you closing the SSH session
#    (the script prints the exact nohup command at the end of step 1, or
#    just ssh in yourself):
ssh -p <PORT> root@<HOST>
cd /workspace
nohup python3 -m pi_tgat_mappo.train \
    --n-min 20 --n-max 50 --max-steps 1000 --iterations 500 \
    > train.log 2>&1 &
disown
exit

# 3. Whenever you want results (during or after training):
./fetch_results.sh <HOST> <PORT>
```

`config.py`'s `resolve_device()` already auto-detects CUDA (it checks
`mps` first, then `cuda`, then falls back to `cpu`) -- no code changes are
needed to use the GPU; on a Linux instance `mps` is never available, so it
lands on `cuda` automatically.

## What actually gets faster on a 4090, and what doesn't

The GAT/GRU matrix operations move to the GPU and will be substantially
faster than the CPU/MPS runs you did locally, and you have far more VRAM
headroom (24 GB) than the earlier disk-constrained sandbox ever had --
comfortably enough to run at the paper's real N in [20, 50] and larger
hidden dims if you want to go beyond Table 1's 128.

What does **not** parallelize automatically: `mappo.py`'s recurrent
update replays each episode's timesteps in a plain Python `for t in
range(T)` loop (full-episode backprop-through-time through the GRU), and
episode collection is similarly sequential. That loop is inherently
serial regardless of GPU, so a bigger GPU shortens each step's compute but
doesn't remove the step count -- expect a large speedup over CPU, but not
a linear one with `--max-steps`. If you want to push toward the paper's
actual 6000-step, 1e7-total-step scale, the highest-leverage next change
would be batching multiple episodes' timesteps together on the GPU (or
truncated BPTT with a shorter window) rather than raising `--iterations`
alone -- happy to help with that once you've confirmed the current design
trains correctly at a moderate scale.

## Cost reminder

Vast.ai bills hourly while the instance is up, independent of whether a
job is running. Destroy or stop the instance from
https://cloud.vast.ai/instances/ when you're done for the session.
