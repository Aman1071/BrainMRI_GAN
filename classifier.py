import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import ResNet50
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

class EnhancedBrainTumorClassifier:
    def __init__(self, use_transfer_learning=False):
        self.use_tl = use_transfer_learning
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        inp = layers.Input(shape=(128,128,1))
        if self.use_tl:
            rgb = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inp)
            base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            for l in base.layers[:-30]: 
                l.trainable = False
            x = base(rgb)
        else:
            x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(inp)
            x = layers.MaxPooling2D()(x); x = layers.BatchNormalization()(x)
            # Attention mechanism
            att = layers.Conv2D(1,(1,1),activation='sigmoid')(x)
            x = layers.Multiply()([x, att])
            x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256,activation='relu')(x)
        out = layers.Dense(1,activation='sigmoid')(x)
        
        m = models.Model(inp, out)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', Precision(name='prec'), Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])
        return m

    def prepare_data(self, tumor, no_tumor, test_size=0.2, val_size=0.1):
        """Prepare train/val/test splits with proper stratification."""
        y_t, y_n = np.ones(len(tumor)), np.zeros(len(no_tumor))
        X = np.vstack([tumor, no_tumor])
        y = np.hstack([y_t, y_n])
        X = (X + 1) / 2  # Normalize to [0,1]
        
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_tr, y_tr, X_val, y_val, epochs=100, save_path='models'):
        """Train with advanced callbacks."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping('val_auc', patience=15, restore_best_weights=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=8, min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint(f"{save_path}/best_classifier.h5", 
                                             monitor='val_auc', save_best_only=True, mode='max')
        ]
        
        # Calculate class weights for imbalanced datasets
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        self.history = self.model.fit(
            X_tr, y_tr, 
            validation_data=(X_val, y_val),
            epochs=epochs, 
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        return self.history

    def evaluate(self, X, y, save_path='models'):
        """Comprehensive evaluation with visualizations."""
        # Get predictions
        y_pred_prob = self.model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_pred_prob = y_pred_prob.flatten()
        
        # Calculate metrics
        loss, acc, prec, rec, auc_score = self.model.evaluate(X, y, verbose=0)
        
        print(f"Test Results:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc_score:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y, y_pred, target_names=['No Tumor', 'Tumor']))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y, y_pred, save_path)
        
        # Plot ROC curve
        self._plot_roc_curve(y, y_pred_prob, save_path)
        
        return acc, prec, rec, auc_score

    def _plot_confusion_matrix(self, y_true, y_pred, save_path):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Tumor', 'Tumor'],
                   yticklabels=['No Tumor', 'Tumor'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300)
        plt.close()

    def _plot_roc_curve(self, y_true, y_pred_prob, save_path):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curve.png', dpi=300)
        plt.close()

    def plot_training_history(self, save_path='models'):
        """Plot comprehensive training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(self.history.history['auc'], label='Training')
        axes[1, 0].plot(self.history.history['val_auc'], label='Validation')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision and Recall
        axes[1, 1].plot(self.history.history['prec'], label='Precision (Train)')
        axes[1, 1].plot(self.history.history['val_prec'], label='Precision (Val)')
        axes[1, 1].plot(self.history.history['recall'], label='Recall (Train)')
        axes[1, 1].plot(self.history.history['val_recall'], label='Recall (Val)')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/classifier_training_history.png', dpi=300)
        plt.close()
