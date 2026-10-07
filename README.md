# Composed Image Retrieval (Fashion-IQ)

This repository trains a composed image retrieval model on Fashion-IQ. A query is a reference image plus a short modification caption. The model ranks gallery images so the edited target is near the top.

The model uses SigLIP-SO400M (vision and text) together with ResNet-50 low-level features, a visual token compressor, a target aggregator, composition fusion, and a query residual. Training uses category-aware hard-negative mining and reports Recall@10 and Recall@50.

Run the notebook `main.ipynb`. It expects a CUDA GPU.

## Install

On Linux, install OpenMP first (required by FAISS):

```bash
sudo apt-get install -y libomp-dev
```

Then install Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

`faiss-gpu-cu12` needs a CUDA 12 GPU environment. Install a matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) if `pip install torch` does not see your GPU.

Set `HF_TOKEN` and `wandb_API` in the script to `"API here"` placeholders before downloading SigLIP or logging runs.

## Dataset layout

Point `Config` at the Fashion-IQ root. Defaults in `cir_63.py`:

- `fashioniq_root`: `root/data`
- `image_dir`: `root/data/images`
- `output_dir`: `root/data/checkpoints`

```text
root/data/
  images/
    B0084Y8XIU.jpg
    ...
  captions/
    cap.dress.train.json
    cap.dress.val.json
    cap.shirt.train.json
    cap.shirt.val.json
    cap.toptee.train.json
    cap.toptee.val.json
  image_splits/
    split.dress.train.json
    split.dress.val.json
    split.shirt.train.json
    split.shirt.val.json
    split.toptee.train.json
    split.toptee.val.json
  checkpoints/
```

Images are `{id}.jpg`, `.png`, or `.jpeg`.

Caption files are a list of triplets:

```json
{
  "candidate": "B005X4PL1G",
  "target": "B0084Y8XIU",
  "captions": [
    "is shiny and silver with shorter sleeves",
    "fit and flare"
  ]
}
```

`candidate` is the reference image. `target` is the image to retrieve. `captions` are the two modification sentences. Split files are a JSON list of image ids for that category and split.

Training captions are randomly reordered and joined. Validation captions stay fixed as `"caption0. caption1"`.

## Output of `main.ipynb`

While training, the script prints the resolved config, a parameter summary, per-epoch loss (total, contrastive, triplet), and evaluation metrics every `eval_every` epochs (default 2) plus the last epoch. Metrics are per category and averaged:

- `dress_R@10`, `shirt_R@10`, `toptee_R@10`, `avg_R@10`
- `dress_R@50`, `shirt_R@50`, `toptee_R@50`, `avg_R@50`

Every 5 epochs, and again at the end, it plots loss, Recall@10 / Recall@50, and learning rate.

Files written under `output_dir`:

| File | When | Contents |
|---|---|---|
| `best_model.pt` | Mean validation R@10 improves | Epoch, full `model_state_dict`, metrics |
| `final_model.pt` | After training | Epoch, weights with SigLIP keys removed, optimizer, scheduler, metrics, config |

After training, the script reloads `best_model.pt` and prints `TEST RESULTS` (the same val Recall@10 / Recall@50 numbers) plus the best R@10.

`retrieve` and `show_retrieval` are defined at the end of the file. They are not called automatically. `retrieve` returns the top-k `(image_id, similarity)` pairs for one reference image and modification text. `show_retrieval` plots the reference next to those results.
