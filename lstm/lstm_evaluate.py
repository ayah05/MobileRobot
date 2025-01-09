import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from lstm_train import main as train_main
import seaborn as sns


def evaluate_predictions(model, test_loader, device):
    """Evaluate model predictions with multiple metrics"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs, _ = model(features)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.squeeze(1).cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Separate position and orientation
    pos_pred = all_predictions[:, :2]  # x, y only
    pos_true = all_targets[:, :2]
    orient_pred = all_predictions[:, 2:]  # quaternion
    orient_true = all_targets[:, 2:]

    # Calculate metrics
    pos_mse = mean_squared_error(pos_true, pos_pred)
    pos_rmse = np.sqrt(pos_mse)
    pos_mae = mean_absolute_error(pos_true, pos_pred)

    orient_mse = mean_squared_error(orient_true, orient_pred)
    orient_rmse = np.sqrt(orient_mse)
    orient_mae = mean_absolute_error(orient_true, orient_pred)

    # Calculate per-axis position errors
    axis_rmse = np.sqrt(np.mean((pos_true - pos_pred) ** 2, axis=0))

    metrics = {
        'predictions': all_predictions,
        'targets': all_targets,
        'pos_rmse': pos_rmse,
        'pos_mae': pos_mae,
        'orient_rmse': orient_rmse,
        'orient_mae': orient_mae,
        'axis_rmse': axis_rmse
    }

    return metrics


def plot_training_history():
    """Plot training and validation loss curves"""
    history = torch.load('training_history.pt')
    train_losses = history['train_losses']
    val_losses = history['val_losses']

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./evaluation/training_history.png', dpi=300)
    plt.close()


def plot_trajectory_comparison(predictions, targets, sample_range=range(100)):
    """Plot 2D trajectory comparison"""
    plt.figure(figsize=(15, 5))

    # 2D trajectory plot
    plt.subplot(1, 2, 1)
    plt.plot(targets[sample_range, 0], targets[sample_range, 1],
             label='True Trajectory', linewidth=2)
    plt.plot(predictions[sample_range, 0], predictions[sample_range, 1],
             '--', label='Predicted Trajectory', linewidth=2)
    plt.title('2D Trajectory Comparison', fontsize=12)
    plt.xlabel('X Position (m)', fontsize=10)
    plt.ylabel('Y Position (m)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Position error over time
    plt.subplot(1, 2, 2)
    pos_error = np.sqrt(np.sum((predictions[sample_range, :2] -
                                targets[sample_range, :2]) ** 2, axis=1))
    plt.plot(pos_error, linewidth=2)
    plt.title('Position Error Over Time', fontsize=12)
    plt.xlabel('Time Step', fontsize=10)
    plt.ylabel('Error (m)', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./evaluation/trajectory_comparison.png', dpi=300)
    plt.close()


def plot_detailed_error_analysis(predictions, targets):
    """Enhanced error analysis with separate position and orientation metrics"""
    fig = plt.figure(figsize=(15, 10))

    # Position errors
    ax1 = plt.subplot(2, 2, 1)
    pos_errors = predictions[:, :2] - targets[:, :2]
    sns.boxplot(data=pd.DataFrame(pos_errors, columns=['X', 'Y']), ax=ax1)
    plt.title('Position Errors by Axis', fontsize=12)
    plt.ylabel('Error (m)', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Orientation errors (quaternion components)
    ax2 = plt.subplot(2, 2, 2)
    orient_errors = predictions[:, 2:] - targets[:, 2:]
    sns.boxplot(data=pd.DataFrame(orient_errors,
                                  columns=['qx', 'qy', 'qz', 'qw']), ax=ax2)
    plt.title('Orientation Errors', fontsize=12)
    plt.ylabel('Error (quaternion)', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Error distribution
    ax3 = plt.subplot(2, 2, 3)
    pos_error_mag = np.sqrt(np.sum(pos_errors ** 2, axis=1))
    sns.histplot(pos_error_mag, bins=50, ax=ax3)
    plt.title('Position Error Magnitude Distribution', fontsize=12)
    plt.xlabel('Error (m)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Correlation plot
    ax4 = plt.subplot(2, 2, 4)
    sns.scatterplot(x=targets[:, 0], y=predictions[:, 0], alpha=0.5, ax=ax4)
    plt.plot([targets[:, 0].min(), targets[:, 0].max()],
             [targets[:, 0].min(), targets[:, 0].max()],
             'r--', label='Perfect Prediction')
    plt.title('X Position: Predicted vs True', fontsize=12)
    plt.xlabel('True Position (m)', fontsize=10)
    plt.ylabel('Predicted Position (m)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('./evaluation/detailed_error_analysis.png', dpi=300)
    plt.close()


def calculate_trajectory_metrics(predictions, targets):
    """Calculate additional trajectory-specific metrics"""
    # Average trajectory deviation
    traj_deviation = np.mean(np.sqrt(np.sum((predictions[:, :2] -
                                             targets[:, :2]) ** 2, axis=1)))

    # Maximum position error
    max_pos_error = np.max(np.sqrt(np.sum((predictions[:, :2] -
                                           targets[:, :2]) ** 2, axis=1)))

    # Orientation stability (quaternion dot product)
    quat_stability = np.mean(np.abs(np.sum(predictions[:, 2:] *
                                           targets[:, 2:], axis=1)))

    return {
        'avg_trajectory_deviation': traj_deviation,
        'max_position_error': max_pos_error,
        'quaternion_stability': quat_stability
    }


def print_metrics_summary(metrics, trajectory_metrics):
    """Print comprehensive metrics summary"""
    print("\nPosition Prediction Metrics:")
    print("-" * 30)
    print(f"Position RMSE: {metrics['pos_rmse']:.4f} m")
    print(f"Position MAE: {metrics['pos_mae']:.4f} m")
    print("\nPer-Axis RMSE:")
    print(f"X-axis: {metrics['axis_rmse'][0]:.4f} m")
    print(f"Y-axis: {metrics['axis_rmse'][1]:.4f} m")

    print("\nOrientation Prediction Metrics:")
    print("-" * 30)
    print(f"Orientation RMSE: {metrics['orient_rmse']:.4f}")
    print(f"Orientation MAE: {metrics['orient_mae']:.4f}")

    print("\nTrajectory Metrics:")
    print("-" * 30)
    print(f"Average Trajectory Deviation: {trajectory_metrics['avg_trajectory_deviation']:.4f} m")
    print(f"Maximum Position Error: {trajectory_metrics['max_position_error']:.4f} m")
    print(f"Quaternion Stability: {trajectory_metrics['quaternion_stability']:.4f}")


def main():
    # Train the model and get necessary components
    model, test_loader, device = train_main()

    # Load best model weights
    model.load_state_dict(torch.load('best_model.pt'))

    # Evaluate predictions
    metrics = evaluate_predictions(model, test_loader, device)
    trajectory_metrics = calculate_trajectory_metrics(
        metrics['predictions'], metrics['targets']
    )

    # Generate visualizations
    plot_training_history()
    plot_trajectory_comparison(metrics['predictions'], metrics['targets'])
    plot_detailed_error_analysis(metrics['predictions'], metrics['targets'])

    # Print metrics summary
    print_metrics_summary(metrics, trajectory_metrics)


if __name__ == "__main__":
    main()