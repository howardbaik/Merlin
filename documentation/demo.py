"""
Download Merlin and test the model on sample data downloaded from Hugging Face.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from merlin import Merlin
from merlin.data import DataLoader, download_sample_data

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = os.path.join(os.getcwd(), "abct_data")
cache_dir = data_dir.replace("abct_data", "abct_data_cache")

datalist = [
    {
        "image": download_sample_data(data_dir),  # returns local path to nifti file
        "text": (
            "Lower thorax: A small low-attenuating fluid structure is noted in the right cardiophrenic angle "
            "in keeping with a tiny pericardial cyst."
            "Liver and biliary tree: Normal. Gallbladder: Normal. Spleen: Normal. Pancreas: Normal. "
            "Adrenal glands: Normal. "
            "Kidneys and ureters: Symmetric enhancement and excretion of the bilateral kidneys, with no "
            "striated nephrogram to suggest pyelonephritis. Urothelial enhancement bilaterally, consistent "
            "with urinary tract infection. No renal/ureteral calculi. No hydronephrosis. "
            "Gastrointestinal tract: Normal. Normal gas-filled appendix. Peritoneal cavity: No free fluid. "
            "Bladder: Marked urothelial enhancement consistent with cystitis. Uterus and ovaries: Normal. "
            "Vasculature: Patent. Lymph nodes: Normal. Abdominal wall: Normal. "
            "Musculoskeletal: Degenerative change of the spine."
        ),
    },
]

dataloader = DataLoader(
    datalist=datalist,
    cache_dir=cache_dir,
    batchsize=8,
    shuffle=True,
    num_workers=0,
)


# --- Default: contrastive embeddings + phenotype predictions ---

model = Merlin()
model.eval()
model.cuda()

for batch in dataloader:
    outputs = model(batch["image"].to(device), batch["text"])
    print("\n================== Output Shapes ==================")
    print(f"Contrastive image embeddings shape: {outputs[0].shape}")
    print(f"Phenotype predictions shape:        {outputs[1].shape}")
    print(f"Contrastive text embeddings shape:  {outputs[2].shape}")


# --- Image embeddings ---

model = Merlin(ImageEmbedding=True)
model.eval()
model.cuda()

for batch in dataloader:
    print("Passed Once")
    outputs = model(batch["image"].to(device))
    print("\n================== Output Shapes ==================")
    print(f"Image embeddings shape (Can be used for downstream tasks): {outputs[0].shape}")


# --- Phenotype predictions ---
model = Merlin(PhenotypeCls=True)
model.eval()
model.cuda()

phenotypes = pd.read_csv(os.path.join(os.getcwd(), "documentation", "phenotypes.csv"))

for batch in dataloader:
    outputs = model(batch["image"].to(device)).squeeze(0).detach().cpu().numpy()

    top_indices = np.argsort(outputs)[::-1][:3]
    top_probs = outputs[top_indices]
    top_phenos = phenotypes.iloc[top_indices].values  # first col = phenotype names

    console = Console()
    print("\nTop 3 predicted phenotypes:")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Phencode", style="cyan", width=10)
    table.add_column("Phecode Description", style="green", max_width=30, overflow="fold")
    table.add_column("Probability", style="yellow")

    for pheno_row, prob in zip(top_phenos, top_probs):
        table.add_row(str(pheno_row[0]), pheno_row[1], f"{prob:.4f}")

    console.print(table)


# --- Five-year disease prediction ---

model = Merlin(FiveYearPred=True)
model.eval()
model.cuda()

disease_names = [
    "Cardiovascular Disease (CVD)",
    "Ischemic Heart Disease (IHD)",
    "Hypertension (HTN)",
    "Diabetes Mellitus (DM)",
    "Chronic Kidney Disease (CKD)",
    "Chronic Liver Disease (CLD)",
]

for batch in dataloader:
    outputs = model(batch["image"].to(device)).squeeze(0)

    console = Console()
    console.print("\nFive year disease prediction probabilities:", style="bold magenta")

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Disease", style="cyan", width=25)
    table.add_column("Probability", style="yellow", justify="right")

    for disease, prob in zip(disease_names, outputs):
        table.add_row(disease, f"{prob:.4f}")

    console.print(table)
