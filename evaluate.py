
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config_sample as config

def _parse_function(example_proto):
    features = {
        'x_input': tf.io.FixedLenFeature([], tf.string),
        'x_output': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    x_input = tf.io.decode_raw(parsed_features['x_input'], tf.float32)
    x_output = tf.io.decode_raw(parsed_features['x_output'], tf.float32)
    x_input = tf.reshape(x_input, config.config['input_shape'])
    x_output = tf.reshape(x_output, config.config['output_shape'])
    return x_input, x_output

def get_dataset(tfrecords, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    return dataset

def plot_example(input_img, pred_img, true_img, idx, out_dir):
    error_img = np.abs(pred_img - true_img)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(input_img.squeeze(), cmap='viridis')
    axs[0].set_title('Input')

    axs[1].imshow(pred_img.squeeze(), cmap='viridis')
    axs[1].set_title('Prediction')

    axs[2].imshow(true_img.squeeze(), cmap='viridis')
    axs[2].set_title('Ground Truth')

    axs[3].imshow(error_img.squeeze(), cmap='inferno')
    axs[3].set_title('|Prediction - Truth|')

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"comparison_{idx:03d}.png"))
    plt.close()

def main():
    model_path = os.path.join(config.config['save_dir'], 'saved_model.h5')
    out_dir = os.path.join(config.config['save_dir'], 'evaluation_results')
    os.makedirs(out_dir, exist_ok=True)

    model = tf.keras.models.load_model(model_path, compile=False)

    dataset = get_dataset(config.config['val_tfrecords'], batch_size=1)

    all_preds, all_trues = [], []
    for idx, (x, y_true) in enumerate(dataset):
        y_pred = model.predict(x)
        all_preds.append(y_pred[0])
        all_trues.append(y_true[0])
        if idx < 10:
            plot_example(x[0].numpy(), y_pred[0], y_true[0].numpy(), idx, out_dir)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    mse = np.mean((all_preds - all_trues) ** 2)
    mae = np.mean(np.abs(all_preds - all_trues))

    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(f"MSE: {mse:.6f}\nMAE: {mae:.6f}\n")

    print(f"✅ Evaluation done. MSE: {mse:.6f}, MAE: {mae:.6f}")
    print(f"📁 Results saved to {out_dir}")

if __name__ == "__main__":
    main()
