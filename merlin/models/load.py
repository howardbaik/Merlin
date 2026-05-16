import os
from typing import Dict, Any

import torch
from torch import nn

from merlin.models.build import MerlinArchitecture
from merlin.models.radiology_report_generation import Clip3DForTextGeneration
from merlin.utils import download_file


REPO_ID = "stanfordmimi/Merlin"

# Model configurations for each supported task.
# Each entry maps a task name to its model builder class and checkpoint filename.
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
    },
    "report_generation": {
        "builder": Clip3DForTextGeneration,
        "checkpoint": "resnet_gpt2_best_stanford_report_generation_average.pt",
    },
    "five_year_disease_prediction": {
        "builder": MerlinArchitecture,
        "checkpoint": "resnet_clinical_longformer_five_year_disease_prediction.pt",
    },
}


class Merlin(nn.Module):
    def __init__(
        self,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        RadiologyReport: bool = False,
        FiveYearPred: bool = False,
    ):
        super().__init__()

        if sum([ImageEmbedding, PhenotypeCls, FiveYearPred]) > 1:
            raise ValueError(
                "ImageEmbedding, PhenotypeCls, and FiveYearPred cannot be True at the same time."
            )

        if RadiologyReport:
            self.task = "report_generation"
        elif FiveYearPred:
            self.task = "five_year_disease_prediction"
        else:
            self.task = "default"

        self._config = MODEL_CONFIGS[self.task]

        model_kwargs = (
            {}
            if RadiologyReport
            else {
                "ImageEmbedding": ImageEmbedding,
                "PhenotypeCls": PhenotypeCls,
                "FiveYearPred": FiveYearPred,
            }
        )
        self.model = self._load_model(**model_kwargs)

    def _load_model(self, **kwargs) -> nn.Module:
        """Downloads the correct checkpoint and constructs the appropriate model."""
        checkpoint_name = self._config["checkpoint"]
        model_builder = self._config["builder"]

        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        self._download_checkpoint(filename=checkpoint_name, local_dir=local_dir)

        model = model_builder(**kwargs)
        checkpoint_path = os.path.join(local_dir, checkpoint_name)

        print(f"Loading checkpoint for '{self.task}' task from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        if self.task == "five_year_disease_prediction":
            model.encode_image.i3_resnet.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict)

        return model

    def _download_checkpoint(self, filename: str, local_dir: str):
        if not os.path.exists(os.path.join(local_dir, filename)):
            print(f"Downloading {filename} from Hugging Face Hub...")
            download_file(repo_id=REPO_ID, filename=filename, local_dir=local_dir)

    def forward(self, *args, **kwargs):
        """Delegates the forward call to the underlying model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generates text if the model is in RadiologyReport mode.
        Passes all arguments to the underlying model's generate method.
        """
        if self.task != "report_generation":
            raise AttributeError(
                "The 'generate' method is only available when RadiologyReport=True."
            )
        return self.model.generate(*args, **kwargs)
