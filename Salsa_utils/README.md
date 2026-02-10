# Salsa_utils

Utilities and extensions for the Duolando baseline. More features will be added here over time.

## Visualization app

A Gradio app to load DD100 data, visualize ground-truth duet motion, and run inference with the pretrained Follower GPT and Follower GPT w. RL models.

**Run from the Duolando repo root:**

```bash
cd Baselines/Salsa_Duolando
python Salsa_utils/visualization/vis_app.py
```

Optional: `--port 7861` if 7860 is in use, `--share` for a public link.
