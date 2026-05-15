import os
import pickle
import random
import sys

import torch
import torchvision
import absl.flags
import absl.app
from tqdm import tqdm

from src import segmentations
from src import model_utils
from src import mask_utils
from src import activation_utils
from src import algorithms
from src import utils
from src import formula as F
from src import settings
from src import constants as C

# User flags
absl.flags.DEFINE_string("subset", "ade20k", "subset to use. Values:[ade20k, pascal]")
absl.flags.DEFINE_string("model", "resnet18", "model to use.")
absl.flags.DEFINE_string("pretrained", "places365", "pretrained weights.")
absl.flags.DEFINE_string("device", "cuda", "device to use")
absl.flags.DEFINE_boolean("pre_load_masks", False, "whether to pre-load masks")
absl.flags.DEFINE_integer("length", 3, "length of explanations")
absl.flags.DEFINE_integer("num_clusters", 5, "number of clusters")
absl.flags.DEFINE_integer("beam_limit", 5, "beam limit")
absl.flags.DEFINE_integer("random_units", 0, "number of units")

# NEW TASK 7 FLAG: Allow user to specify custom layers
absl.flags.DEFINE_list("target_layers", ["layer4"], "Comma-separated list of layers to analyze")

absl.flags.DEFINE_string("root_models", "data/model/", "root directory for models")
absl.flags.DEFINE_string("root_datasets", "data/dataset/", "root directory for datasets")
absl.flags.DEFINE_string("root_segmentations", "data/cache/segmentations/", "root directory for segmentations")
absl.flags.DEFINE_string("root_activations", "data/cache/activations/", "root directory for activations")
absl.flags.DEFINE_string("root_results", "data/results/", "root directory for results")
absl.flags.DEFINE_integer("seed", 0, "seed to use")

FLAGS = absl.flags.FLAGS

def main(argv):
    if FLAGS.num_clusters < 1:
        raise ValueError("num_clusters must be greater than 0")
    
    generator = utils.set_seed(FLAGS.seed)
    target_unit = int(os.environ.get("TARGET_UNIT", "-1"))

    cfg = settings.Settings(
        subset=FLAGS.subset, model=FLAGS.model, pretrained=FLAGS.pretrained,
        num_clusters=FLAGS.num_clusters, beam_limit=FLAGS.beam_limit,
        device=FLAGS.device, root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets, root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations, root_results=FLAGS.root_results
    )
    sparse_segmentation_directory = cfg.get_segmentation_directory()
    mask_shape = cfg.get_mask_shape()

    dataset = segmentations.BrodenDataset(
        cfg.dir_datasets, subset=cfg.index_subset, resolution=cfg.get_img_size(),
        broden_version=1, transform_image=torchvision.transforms.Compose([
            torchvision.transforms.Resize(cfg.get_img_size()),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(cfg.get_image_mean(), cfg.get_image_stdev()),
        ]),
    )
    segmentation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=C.BATCH_SIZE, worker_init_fn=utils.seed_worker, generator=generator,
    )

    model = model_utils.load_model_from_settings(cfg, device=cfg.device)
    masks = mask_utils.get_masks(sparse_segmentation_directory, segmentation_loader, dataset.labels, cfg.device, pre_load=FLAGS.pre_load_masks)
    masks_info = mask_utils.get_masks_info(masks, config=cfg)

    # Loop over user-specified layers instead of hardcoding
    for layer_name in FLAGS.target_layers:
        if layer_name not in cfg.get_feature_names():
            print(f"Warning: {layer_name} is not a valid feature name. Skipping.")
            continue

        num_units = model_utils.get_number_of_units(model, layer_name, cfg)
        activations = model_utils.get_layer_activations(segmentation_loader, model, layer_name, range(num_units), cfg.get_activation_directory())

        if target_unit != -1:
            selected_units = [target_unit]
        else:
            # For random units if not using the Bash loop
            selected_units = random.sample(range(num_units), 10) if FLAGS.random_units == 0 else range(10)

        for unit in tqdm(selected_units, desc=f"Explaining units in {layer_name}"):
            unit_activations = activations[unit]
            activation_ranges = activation_utils.compute_activation_ranges(unit_activations, FLAGS.num_clusters)

            for cluster_index, activation_range in enumerate(sorted(activation_ranges)):
                # ONLY process/print the highest cluster (Cluster 4) to save time
                if cluster_index != (FLAGS.num_clusters - 1):
                    continue

                dir_current_results = f"{cfg.get_results_directory()}/{layer_name}/{unit}/{activation_range}"
                os.makedirs(dir_current_results, exist_ok=True)
                file_algo_results = f"{dir_current_results}/{FLAGS.length}.pickle"
                
                if not os.path.exists(file_algo_results):
                    bitmaps = activation_utils.compute_bitmaps(unit_activations, activation_range, mask_shape=mask_shape).to(cfg.device)
                    best_label, best_iou, visited = algorithms.get_heuristic_scores(
                        masks, bitmaps, segmentations_info=masks_info, heuristic="mmesh",
                        length=FLAGS.length, max_size_mask=cfg.get_max_mask_size(),
                        mask_shape=cfg.get_mask_shape(), device=cfg.device,
                    )
                    with open(file_algo_results, "wb") as file:
                        pickle.dump((best_label, best_iou, visited), file)
                else:
                    with open(file_algo_results, "rb") as file:
                        best_label, best_iou, visited = pickle.load(file)

                string_label = F.get_formula_str(best_label, dataset.labels)
                
                # TASK 7 DELIVERABLE FORMAT: <Layer> | <Unit ID> - <Explanation> - <Highest Cluster>
                print(f"RESULT: {layer_name} | Unit {unit} - {string_label} - Cluster {cluster_index}")

        if target_unit != -1:
            sys.exit(0)

if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
