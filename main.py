import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from data_loader import BrainTumorDataLoader
from dcgan_brain_tumor import BrainTumorDCGAN
from classifier import EnhancedBrainTumorClassifier

def progressive_training(dl, save_path):
    resolutions = [32, 64, 128]
    for res in resolutions:
        print(f"\n=== Training GAN at {res}×{res} ===")
        info = dl.create_progressive_datasets([res])[res]
        ds = info['tumor_ds']
        gan = BrainTumorDCGAN(
            start_res=res,
            latent_dim=100,
            use_wgan_gp=True
        )
        gan.train(ds, epochs=50, save_interval=10, 
                 save_path=f"{save_path}/tumor_{res}")
    return gan

def plot_comprehensive_comparison(hist_orig, hist_aug, save_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(hist_orig.history['val_accuracy'], label='Original Val')
    plt.plot(hist_aug.history['val_accuracy'], label='Augmented Val')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'val_accuracy_comparison.png'))
    plt.close()

def main():
    DATA = 'brain_tumor_dataset'
    SYN = os.path.join(DATA, 'synthetic_images')
    MOD = os.path.join(DATA, 'models')
    os.makedirs(SYN, exist_ok=True)
    os.makedirs(MOD, exist_ok=True)
    
    print("=== Enhanced Brain Tumor DCGAN Training ===")
    print("Features: Progressive Growing, WGAN-GP, Medical Augmentation")
    
    dl = BrainTumorDataLoader(DATA, use_medical_augmentation=True)
    use_progressive = input("Use progressive training? (y/n): ").lower() == 'y'
    
    if use_progressive:
        print("Starting progressive training...")
        gan = progressive_training(dl, SYN)
    else:
        print("Starting standard 128x128 training...")
        ds_t, _, t_imgs, _ = dl.get_datasets()
        gan = BrainTumorDCGAN(start_res=128, latent_dim=100, use_wgan_gp=True)
        gan.train(ds_t, epochs=200, save_interval=20, 
                 save_path=os.path.join(SYN, 'tumor_128'))
    
    # Generate synthetic images - CORRECTED SECTION
    X_t, X_n = dl.load_dataset()
    count = min(len(X_t), len(X_n))
    noise = tf.random.normal([count, gan.latent_dim])  # Use gan.latent_dim
    synthetic_tumor = gan.gen_model(noise, training=False)  # Use gen_model
    
    # Classifier training and evaluation
    clf_orig = EnhancedBrainTumorClassifier(use_transfer_learning=False)
    X_tr1, X_val1, X_te1, y_tr1, y_val1, y_te1 = clf_orig.prepare_data(X_t, X_n)
    hist_orig = clf_orig.train(X_tr1, y_tr1, X_val1, y_val1, save_path=MOD)
    
    clf_aug = EnhancedBrainTumorClassifier(use_transfer_learning=True)
    X_tr2, X_val2, X_te2, y_tr2, y_val2, y_te2 = clf_aug.prepare_data(
        np.vstack([X_t, synthetic_tumor.numpy()]), 
        np.vstack([X_n, synthetic_tumor.numpy()])
    )
    hist_aug = clf_aug.train(X_tr2, y_tr2, X_val2, y_val2, save_path=MOD)
    
    # Plot results
    plot_comprehensive_comparison(hist_orig, hist_aug, MOD)
    
    # Save and print results
    results = {
        'original_accuracy': hist_orig.history['val_accuracy'][-1],
        'augmented_accuracy': hist_aug.history['val_accuracy'][-1],
        'improvement': hist_aug.history['val_accuracy'][-1] - hist_orig.history['val_accuracy'][-1]
    }
    with open(os.path.join(MOD, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n===== Final Results =====")
    print(f"Original Accuracy: {results['original_accuracy']:.4f}")
    print(f"Augmented Accuracy: {results['augmented_accuracy']:.4f}")
    print(f"Improvement: {results['improvement']:.4f}")

if __name__ == "__main__":
    main()
