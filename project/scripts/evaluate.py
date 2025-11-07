"""
Evaluation script.

This is a template for the evaluation script. Adapt it to your specific needs.
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import PhysicallyInformedAE
from data.dataset import SpectraDataset
from evaluation.metrics import compute_mae, compute_rmse, compute_r2_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate PhysicallyInformedAE model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    return parser.parse_args()


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on dataset.

    Args:
        model: PyTorch Lightning model.
        dataloader: DataLoader for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            noisy_spectra = batch['noisy_spectra'].to(device)
            params_true = batch['params'].to(device)

            # Forward pass
            output = model(noisy_spectra)
            params_pred = output['params']

            all_predictions.append(params_pred.cpu())
            all_targets.append(params_true.cpu())

    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = {
        'mae': compute_mae(predictions, targets),
        'rmse': compute_rmse(predictions, targets),
        'r2': compute_r2_score(predictions, targets),
    }

    # Per-parameter metrics
    param_names = ['sig0', 'dsig', 'mf_CH4', 'baseline0', 'baseline1', 'baseline2', 'P', 'T']
    for i, name in enumerate(param_names):
        metrics[f'mae_{name}'] = compute_mae(predictions[:, i], targets[:, i])
        metrics[f'rmse_{name}'] = compute_rmse(predictions[:, i], targets[:, i])

    return metrics, predictions, targets


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = PhysicallyInformedAE.load_from_checkpoint(args.checkpoint)

    # TODO: Create evaluation dataset
    # This should match your training dataset setup
    print("Creating evaluation dataset...")
    # eval_dataset = SpectraDataset(...)

    # eval_loader = DataLoader(
    #     eval_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    # )

    # Evaluate
    # print("Evaluating model...")
    # metrics, predictions, targets = evaluate_model(model, eval_loader, args.device)

    # Print results
    # print("\nEvaluation Results:")
    # print("=" * 50)
    # for key, value in metrics.items():
    #     print(f"{key}: {value:.6f}")

    print("Evaluation complete!")


if __name__ == '__main__':
    main()
