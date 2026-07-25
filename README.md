# NeuroScan XAI — Brain Tumour Detection with Explainable AI

Working implementation of your Phase 1 proposal: **"A Unified XAI Framework
for Interpreting Deep Learning Models in Brain Tumour Detection"** (BCS685).
Custom CNN + Grad-CAM + LRP + SHAP on BraTS 2021 FLAIR MRI, served through
a FastAPI backend and a React diagnostic-console frontend.

**This is a fully working, tested system right now** — trained on a
synthetic stand-in dataset (see "About the demo model" below) because this
build environment can't reach Kaggle to download the real ~6GB BraTS 2021
dataset. Swapping in the real dataset takes two commands (Section 2) and
changes nothing else — same model code, same API, same UI.

---

## 1. Project structure

```
braintumor-xai/
├── backend/
│   ├── app/
│   │   ├── model.py         # BrainTumorCNN (Conv→ReLU→MaxPool ×4 + FC head)
│   │   ├── xai.py           # Grad-CAM, LRP (hand-written), SHAP
│   │   └── main.py          # FastAPI app: /health, /predict
│   ├── train_demo.py        # generates synthetic data + trains a demo model (already run)
│   ├── prepare_brats.py     # converts real BraTS .nii.gz -> labeled PNG slices
│   ├── train_real.py        # trains on the real, prepared BraTS data
│   ├── model_weights.pt     # ALREADY TRAINED demo weights, included
│   ├── demo_data/           # the synthetic slices used to train it
│   └── requirements.txt
└── frontend/
    ├── src/App.jsx           # dashboard UI
    ├── src/App.css           # diagnostic-console design system
    └── (standard Vite React app)
```

## 2. Running it as-is (demo model, works immediately)

```bash
# Backend
cd backend
pip install -r requirements.txt --break-system-packages
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open the printed frontend URL (usually `http://localhost:5173`), upload any
grayscale MRI-like image, click **Run Detection + XAI**, and you'll get a
prediction plus three live heatmaps. Try the images inside
`backend/demo_data/tumor/` and `backend/demo_data/notumor/` first.

## 3. Swapping in the real BraTS 2021 dataset

The sandbox that built this can't reach Kaggle, so you'll do this step
yourself — it's two commands.

**Get the data:**
1. Go to Kaggle and search **"BraTS 2021 Task 1"** (e.g.
   `kaggle.com/datasets/dschettler8845/brats-2021-task1`), or use the
   official BraTS registration at https://www.synapse.org/brats if your
   institution needs the fully licensed version.
2. Either download it locally, or — recommended, since it's large and you
   have no GPU-heavy laptop most likely — open a **Kaggle Notebook**,
   attach the dataset, and run everything there (Kaggle gives free GPU
   time, which matters for step 4).
3. You should end up with one folder per patient, each containing
   `..._flair.nii.gz`, `..._t1.nii.gz`, `..._t1ce.nii.gz`, `..._t2.nii.gz`,
   `..._seg.nii.gz`.

**Prepare and train:**
```bash
cd backend
# edit RAW_DATA_DIR at the top of prepare_brats.py to point at your download
python3 prepare_brats.py     # -> brats_prepared/tumor, brats_prepared/notumor
python3 train_real.py        # trains, prints accuracy/precision/recall/F1, saves model_weights.pt
```

That's it — restart `uvicorn`, and the web app now runs on your real,
BraTS-trained model. `prepare_brats.py` implements exactly the
preprocessing your Phase 1 report already committed to (resize,
intensity normalization, Gaussian + median denoising, cleaning of
corrupted/empty slices); `train_real.py` adds the rotation + horizontal
flip augmentation and a proper stratified train/val/test split, and
reports the full metrics you'll want for your results chapter.

## 4. About the demo model (be upfront about this in your presentation)

The `model_weights.pt` included here was trained on **synthetic**
brain-silhouette images with a randomly placed bright "tumour" blob — not
real MRI data — purely so the full pipeline (CNN → Grad-CAM → LRP → SHAP →
web app) is provably working end-to-end today. It gets ~100% accuracy
because the synthetic task is trivially easy; that number means nothing
about real-world performance and shouldn't be quoted to faculty as a
result. Treat it as a wiring test, not a model result. Your actual
reportable numbers come from `train_real.py` on real BraTS data.

## 5. Implementation notes worth knowing for your defense

- **LRP**: Captum's built-in `LRP` class threw a hook-compatibility error
  with Conv2d layers on this PyTorch version (`RuntimeError: hook
  '_backward_hook_input' has changed the size of value` — a known
  library version mismatch, reproducible even on a minimal 1-conv-layer
  model). Rather than downgrade dependencies and risk other breakage, I
  hand-implemented epsilon-LRP directly (`app/xai.py::lrp`) using the
  standard "gradient trick" formulation from Montavon et al.'s LRP
  overview paper. This is a legitimate, citable approach — plenty of LRP
  implementations are hand-written rather than library calls — and it's
  arguably a stronger talking point in your viva ("we implemented LRP's
  propagation rule directly") than "we called a library function."
- **Grad-CAM** target layer is the last convolutional layer
  (`model.gradcam_target_layer`, 128 channels at 8×8 resolution before
  the final pooling) — standard Grad-CAM placement.
- **SHAP** uses `GradientExplainer` (not `KernelExplainer`), which is
  far faster for CNNs — it needs a small background sample set (drawn
  from `demo_data/` here; use a sample from your training set for the
  real model).
- All three explanation methods run on the **same trained model** and
  the **same input**, which is the whole point of a "unified" XAI
  framework — you can literally show faculty the three heatmaps
  disagreeing or agreeing on the same scan.

## 6. Deploying for your Phase 2 demo

For the live evaluation, the simplest path is: run the backend on your
laptop (or a Kaggle/Colab-exposed endpoint via `ngrok` if you trained
there and want to keep using their GPU), and run the frontend either
locally (`npm run dev`) or built as static files (`npm run build` in
`frontend/`, then serve `frontend/dist/`) and deployed to something free
like Vercel or Netlify, pointing `VITE_API_BASE` at wherever your backend
is reachable.
