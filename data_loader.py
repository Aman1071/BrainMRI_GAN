import os, cv2, numpy as np, tensorflow as tf
from scipy.ndimage import map_coordinates, gaussian_filter

class BrainTumorDataLoader:
    def __init__(self, data_path, img_h=128, img_w=128, batch_size=32, use_medical_augmentation=True):
        self.data_path = data_path
        self.img_h = img_h
        self.img_w = img_w
        self.bs = batch_size
        self.use_medical_augmentation = use_medical_augmentation

    def _enhance(self, img_array):
        img2d = img_array[:,:,0] if img_array.ndim == 3 else img_array
        img_uint8 = ((img2d + 1) * 127.5).astype(np.uint8)
        eq = cv2.equalizeHist(img_uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(eq)
        return (clahe.astype(np.float32) / 127.5) - 1

    def _elastic(self, img2d):
        alpha, sigma = img2d.shape[1]*2, img2d.shape[1]*0.08
        dx = gaussian_filter((np.random.rand(*img2d.shape)*2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*img2d.shape)*2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(img2d.shape[1]), np.arange(img2d.shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
        deformed = map_coordinates(img2d, indices, order=1, mode='reflect').reshape(img2d.shape)
        return deformed

    def _augment(self, img):
        if not self.use_medical_augmentation:
            return img
        img2d = img[:,:,0] if img.ndim == 3 else img
        return np.expand_dims(self._elastic(img2d), axis=-1)

    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = cv2.resize(img, (self.img_w, self.img_h))
        return np.expand_dims((img.astype(np.float32) - 127.5) / 127.5, axis=-1)

    def load_dataset(self):
        tumor, no_tumor = [], []
        for label in ('yes','no'):
            folder = os.path.join(self.data_path, label)
            if os.path.isdir(folder):
                for fname in os.listdir(folder):
                    if fname.lower().endswith(('.png','.jpg','.jpeg')):
                        img = self._load_img(os.path.join(folder, fname))
                        if img is not None:
                            (tumor if label=='yes' else no_tumor).append(img)
        return np.array(tumor), np.array(no_tumor)

    def get_datasets(self):
        tumor_imgs, no_imgs = self.load_dataset()
        print(f"Loaded {len(tumor_imgs)} tumor and {len(no_imgs)} no-tumor images")
        def preprocess(x):
            x_np = x.numpy() if hasattr(x, 'numpy') else x
            img2d = x_np[:,:,0] if x_np.ndim==3 else x_np
            enh = self._enhance(img2d)
            aug = self._augment(enh)
            return tf.reshape(aug, [self.img_h, self.img_w, 1])
        ds_t = tf.data.Dataset.from_tensor_slices(tumor_imgs) \
               .map(lambda x: tf.py_function(preprocess, [x], tf.float32)) \
               .batch(self.bs).prefetch(tf.data.AUTOTUNE)
        ds_n = tf.data.Dataset.from_tensor_slices(no_imgs) \
               .map(lambda x: tf.py_function(preprocess, [x], tf.float32)) \
               .batch(self.bs).prefetch(tf.data.AUTOTUNE)
        return ds_t, ds_n, tumor_imgs, no_imgs

    def create_progressive_datasets(self, resolutions=[32,64,128]):
        orig_h, orig_w = self.img_h, self.img_w
        tumor_imgs, no_imgs = self.load_dataset()
        datasets = {}
        for res in resolutions:
            self.img_h, self.img_w = res, res
            def resize_all(arr):
                return np.array([np.expand_dims(cv2.resize(i[:,:,0], (res,res)), axis=-1) for i in arr])
            t_res, n_res = resize_all(tumor_imgs), resize_all(no_imgs)
            # Ensure correct shape for each dataset
            def validate_shape(x):
                return tf.ensure_shape(x, [res, res, 1])
            tumor_ds = tf.data.Dataset.from_tensor_slices(t_res).map(validate_shape).batch(self.bs).prefetch(tf.data.AUTOTUNE)
            no_tumor_ds = tf.data.Dataset.from_tensor_slices(n_res).map(validate_shape).batch(self.bs).prefetch(tf.data.AUTOTUNE)
            datasets[res] = {
                'tumor_ds': tumor_ds,
                'no_tumor_ds': no_tumor_ds,
                'tumor_imgs': t_res,
                'no_tumor_imgs': n_res
            }
        self.img_h, self.img_w = orig_h, orig_w
        return datasets
