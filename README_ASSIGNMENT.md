# ERA 5 Session 14 – Assignment Completion

## Assignment (from the lecture)

> Train a linear (dense) model and then convert that into an MoE of your own choice.  
> Decide the model size and the data.  
> Show that:  
> 1. It continues to train after conversion  
> 2. The loss actually drops

## What was implemented

| Item | Choice |
|------|--------|
| **Data** | Synthetic multi-class classification (20 k train / 4 k val, 64-dim features, 10 classes) |
| **Dense model** | 3-layer MLP (`64 → 256 → 256 → 10`) |
| **MoE conversion** | Middle layer replaced by MoE |
| **MoE config** | 8 experts, top-k = 2, each expert 128-wide, + shared expert (64-wide) |
| **Training** | Dense for 8 epochs → MoE for 12 epochs |
| **Evidence** | Training continues without error; training loss drops from **0.3705 → 0.0339** |

## Files

- `moe_assignment.py` – complete runnable script
- `moe_assignment_loss_curves.png` – loss curves for both phases
- `moe_model.pt` – final MoE checkpoint + config

## How to run

```bash
python moe_assignment.py
```

(Requires only `torch`, `numpy`, `matplotlib` – all available in the environment.)

## How to submit

1. Run the script (or just submit the already-generated outputs).
2. Include:
   - the script `moe_assignment.py`
   - the loss-curve plot
   - (optional) the checkpoint
3. In the report / comments briefly state:
   - Model size & data choice
   - MoE hyper-parameters (experts=8, top-k=2, shared expert, …)
   - That training continued and training loss dropped (see printed evidence and the plot)

The video did not specify a platform or deadline; use whatever submission method the course already uses for previous assignments.
