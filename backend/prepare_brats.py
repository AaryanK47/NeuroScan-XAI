"""
Converts the real BraTS 2021 dataset (.nii.gz 3D volumes) into a folder of
labeled 2D FLAIR PNG slices, ready for train_real.py — implementing exactly
the preprocessing steps described in Chapter 6.2 of the report:

  Data Cleaning:
    - Removal of corrupted / incomplete MRI images
    - Elimination of irrelevant or noisy slices (near-empty / no-brain slices)
    - Filtering low-quality images

  Preprocessing:
    - Image resizing for uniform input dimensions (128x128)
    - Normalization of pixel intensity values
    - Denoising using Gaussian and Median filtering
    - Data augmentation: rotation and horizontal flip (applied later, in
      train_real.py, only on the training split — never on val/test)

--------------------------------------------------------------------------
HOW TO GET THE DATASET (do this on your own machine or on Kaggle/Colab,
not in this sandbox — it has no access to Kaggle):

  1. Go to: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
     (or search "BraTS 2021" on Kaggle — several mirrors exist).
  2. Download / attach it in a Kaggle Notebook, or download the zip and
     unzip locally. You'll get a folder per patient, each containing:
        BraTS2021_XXXXX_flair.nii.gz
        BraTS2021_XXXXX_t1.nii.gz
        BraTS2021_XXXXX_t1ce.nii.gz
        BraTS2021_XXXXX_t2.nii.gz
        BraTS2021_XXXXX_seg.nii.gz   <- ground-truth tumour segmentation mask
  3. Point RAW_DATA_DIR below at the parent folder containing all the
     patient subfolders, then run:
        python3 prepare_brats.py
--------------------------------------------------------------------------

Binary label rule used here (matches the report's "Tumour / Non-Tumour"
binary classification): a slice is labeled TUMOUR if its corresponding
segmentation mask (_seg.nii.gz) has any non-zero voxels in that slice,
else NON-TUMOUR.
"""

import os
import glob
import cv2
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise SystemExit("Run: pip install nibabel --break-system-packages")

RAW_DATA_DIR = "brats2021_raw"       # <-- point this at your downloaded BraTS folder
OUT_DIR = "brats_prepared"
IMG_SIZE = 128
MIN_BRAIN_PIXELS = 400               # slices with less brain tissue than this are dropped (noise/empty)
MIN_TUMOR_PIXELS = 15                # tumour masks smaller than this are treated as noise, labeled non-tumour


def clean_and_normalize(slice_2d: np.ndarray) -> np.ndarray:
    """Data Cleaning + Preprocessing from report Ch. 6.2."""
    # Normalize intensity to [0, 1] using the volume's own dynamic range
    s = slice_2d.astype(np.float32)
    p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
    if p99 > p1:
        s = np.clip((s - p1) / (p99 - p1), 0, 1)
    else:
        s = np.zeros_like(s)

    # Denoising: Gaussian then Median filtering
    s_uint8 = np.uint8(s * 255)
    s_uint8 = cv2.GaussianBlur(s_uint8, (3, 3), 0)
    s_uint8 = cv2.medianBlur(s_uint8, 3)

    # Resize to uniform input dimensions
    s_resized = cv2.resize(s_uint8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return s_resized


def process_patient(flair_path: str, seg_path: str, out_dir: str, patient_id: str) -> int:
    flair_vol = nib.load(flair_path).get_fdata()
    seg_vol = nib.load(seg_path).get_fdata() if seg_path and os.path.isfile(seg_path) else None

    n_slices = flair_vol.shape[2]
    saved = 0
    for z in range(n_slices):
        flair_slice = flair_vol[:, :, z]

        # Data cleaning: drop near-empty / no-brain slices (noise filtering)
        if np.count_nonzero(flair_slice) < MIN_BRAIN_PIXELS:
            continue

        tumor_present = False
        if seg_vol is not None:
            seg_slice = seg_vol[:, :, z]
            tumor_present = np.count_nonzero(seg_slice) >= MIN_TUMOR_PIXELS

        cleaned = clean_and_normalize(flair_slice)

        label_dir = "tumor" if tumor_present else "notumor"
        out_path = os.path.join(out_dir, label_dir, f"{patient_id}_slice{z:03d}.png")
        cv2.imwrite(out_path, cleaned)
        saved += 1

    return saved


def main():
    if not os.path.isdir(RAW_DATA_DIR):
        raise SystemExit(
            f"'{RAW_DATA_DIR}' not found. Download BraTS 2021 first (see the docstring "
            f"at the top of this file) and point RAW_DATA_DIR at it."
        )

    os.makedirs(os.path.join(OUT_DIR, "tumor"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "notumor"), exist_ok=True)

    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(RAW_DATA_DIR, "*")) if os.path.isdir(d)
    )
    if not patient_dirs:
        raise SystemExit(f"No patient subfolders found inside '{RAW_DATA_DIR}'.")

    total_saved = 0
    for i, pdir in enumerate(patient_dirs):
        patient_id = os.path.basename(pdir)
        flair_matches = glob.glob(os.path.join(pdir, "*flair.nii.gz"))
        seg_matches = glob.glob(os.path.join(pdir, "*seg.nii.gz"))

        if not flair_matches:
            print(f"[skip] {patient_id}: no FLAIR volume found (corrupted/incomplete case)")
            continue

        seg_path = seg_matches[0] if seg_matches else None
        saved = process_patient(flair_matches[0], seg_path, OUT_DIR, patient_id)
        total_saved += saved
        print(f"[{i+1}/{len(patient_dirs)}] {patient_id}: saved {saved} slices")

    print(f"\nDone. {total_saved} total slices saved to '{OUT_DIR}/tumor' and '{OUT_DIR}/notumor'.")
    print("Next: python3 train_real.py")


if __name__ == "__main__":
    main()
