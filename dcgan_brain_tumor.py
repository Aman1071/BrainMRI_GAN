import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow_addons.layers import SpectralNormalization
import os
import cv2

class ProgressiveGenerator:
    def __init__(self, latent_dim=100, start_res=4):
        self.latent_dim = latent_dim
        self.current_res = start_res
        self.blocks = []
        self.rgb_layers = []
        self.build_base()
        
    def build_base(self):
        # Initial block
        input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.current_res*self.current_res*512)(input)
        x = layers.Reshape((self.current_res, self.current_res, 512))(x)
        self.blocks.append(Model(input, x))
        self.rgb_layers.append(layers.Conv2D(1, (1,1), activation='tanh'))
        
    def add_block(self):
        prev_block = self.blocks[-1]
        new_res = self.current_res * 2
        
        # New block
        input = layers.Input(shape=prev_block.output_shape[1:])
        x = layers.UpSampling2D()(input)
        x = SpectralNormalization(layers.Conv2D(512, (3,3), padding='same'))(x)
        x = layers.LeakyReLU(0.2)(x)
        new_block = Model(input, x)
        
        # New RGB layer
        new_rgb = layers.Conv2D(1, (1,1), activation='tanh')
        
        # Update components
        self.blocks.append(new_block)
        self.rgb_layers.append(new_rgb)
        self.current_res = new_res
        
    def build_current(self):
        input = layers.Input(shape=(self.latent_dim,))
        x = self.blocks[0](input)
        for block in self.blocks[1:]:
            x = block(x)
        output = self.rgb_layers[-1](x)
        return Model(input, output)

class ProgressiveDiscriminator:
    def __init__(self, start_res=4):
        self.current_res = start_res
        self.blocks = []
        self.rgb_layers = []
        self.build_base()
        
    def build_base(self):
        # Start with current resolution
        input = layers.Input(shape=(self.current_res, self.current_res, 1))
        x = SpectralNormalization(layers.Conv2D(64, (1,1)))(input)
        self.blocks.append(Model(input, x))
        self.rgb_layers.append(layers.Conv2D(1, (1,1)))
        
    def add_block(self):
        new_res = self.current_res * 2
        
        # New block
        input = layers.Input(shape=(new_res, new_res, 1))
        x = SpectralNormalization(layers.Conv2D(64, (3,3), padding='same'))(input)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.AveragePooling2D()(x)
        new_block = Model(input, x)
        
        # New RGB layer
        new_rgb = layers.Conv2D(1, (1,1))
        
        # Update components
        self.blocks.insert(0, new_block)
        self.rgb_layers.insert(0, new_rgb)
        self.current_res = new_res
        
    def build_current(self):
        input = layers.Input(shape=(self.current_res, self.current_res, 1))
        x = self.rgb_layers[0](input)
        for block in self.blocks:
            x = block(x)
        x = layers.Flatten()(x)
        output = layers.Dense(1)(x)
        return Model(input, output)

class BrainTumorDCGAN:
    def __init__(self, start_res=4, latent_dim=100, use_wgan_gp=True):
        self.latent_dim = latent_dim
        self.use_wgan_gp = use_wgan_gp
        self.current_res = start_res
        
        # Initialize generator and discriminator
        self.generator = ProgressiveGenerator(latent_dim, start_res)
        self.discriminator = ProgressiveDiscriminator(start_res)
        
        # Build models
        self.gen_model = self.generator.build_current()
        self.disc_model = self.discriminator.build_current()
        
        # Optimizers
        self.gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
        self.disc_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5, beta_2=0.9)
    
    def grow_models(self):
        """Grow both generator and discriminator"""
        self.generator.add_block()
        self.discriminator.add_block()
        self.current_res *= 2
        
        # Rebuild models
        self.gen_model = self.generator.build_current()
        self.disc_model = self.discriminator.build_current()
    
    def gradient_penalty(self, real, fake, lam=10.0):
        alpha = tf.random.uniform([tf.shape(real)[0], 1, 1, 1], 0, 1)
        interp = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = self.disc_model(interp, training=True)
        grads = tape.gradient(pred, interp)
        norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        return lam * tf.reduce_mean(tf.square(norms - 1.0))
    
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([tf.shape(images)[0], self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images
            gen_imgs = self.gen_model(noise, training=True)
            
            # Discriminator outputs
            real_output = self.disc_model(images, training=True)
            fake_output = self.disc_model(gen_imgs, training=True)
            
            # Calculate losses
            if self.use_wgan_gp:
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + self.gradient_penalty(images, gen_imgs)
                g_loss = -tf.reduce_mean(fake_output)
            else:
                ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                d_loss = ce(tf.ones_like(real_output), real_output) + ce(tf.zeros_like(fake_output), fake_output)
                g_loss = ce(tf.ones_like(fake_output), fake_output)
        
        # Apply gradients
        gen_grad = gen_tape.gradient(g_loss, self.gen_model.trainable_variables)
        disc_grad = disc_tape.gradient(d_loss, self.disc_model.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_grad, self.gen_model.trainable_variables))
        self.disc_opt.apply_gradients(zip(disc_grad, self.disc_model.trainable_variables))
        return g_loss, d_loss
    
    def train(self, dataset, epochs, save_interval, save_path):
        os.makedirs(save_path, exist_ok=True)
        seed = tf.random.normal([16, self.latent_dim])
        
        for epoch in range(epochs):
            for batch in dataset:
                g_loss, d_loss = self.train_step(batch)
            
            if epoch % save_interval == 0:
                print(f"Epoch {epoch}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
                self._save_images(epoch, seed, save_path)
        
        # Save final models
        self.gen_model.save(os.path.join(save_path, 'generator_final.h5'))
    
    def _save_images(self, epoch, seed, save_path):
        preds = self.gen_model(seed, training=False).numpy()
        for i, img in enumerate(preds):
            # Convert to [0, 255] and save
            img_uint8 = ((img + 1) * 127.5).astype('uint8').squeeze()
            cv2.imwrite(f"{save_path}/epoch_{epoch:03d}_img_{i}.png", img_uint8)
