"""
Neural Amp Model Training System - Production Version
Trains neural networks on SPICE simulation data and/or TINA-collected data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import pickle
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Generator, Any
import os
from pathlib import Path
from collections import OrderedDict
import h5py
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neural_amp_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for neural network training"""
    model_type: str = "lstm"  # "lstm", "cnn", "wavenet", "transformer"
    sequence_length: int = 512
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.1
    sample_rate: int = 44100
    use_spectral_loss: bool = True
    use_time_loss: bool = True
    spectral_loss_weight: float = 0.5
    time_loss_weight: float = 0.5
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    use_mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    use_data_augmentation: bool = True
    augmentation_factor: float = 0.1


class DataGenerator(keras.utils.Sequence):
    """Data generator for efficient batch loading"""
    
    def __init__(self, data_file: str, batch_size: int, sequence_length: int,
                 indices: List[int], shuffle: bool = True):
        self.data_file = data_file
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.indices = indices
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Load data info
        with h5py.File(data_file, 'r') as f:
            self.total_samples = f['input_audio'].shape[0]
    
    def __len__(self):
        """Number of batches per epoch"""
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices for the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Load data
        with h5py.File(self.data_file, 'r') as f:
            X = f['input_audio'][batch_indices]
            y = f['output_audio'][batch_indices]
            controls = f['control_params'][batch_indices]
        
        # Combine input audio with control parameters
        X_combined = np.concatenate([X[..., np.newaxis], controls], axis=-1)
        
        return X_combined, y[..., np.newaxis]
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


class NeuralAmpTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self.training_history = None
        
        # Create output directories
        self.output_dir = Path("neural_amp_models")
        self.output_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Set up TensorFlow
        self.setup_tensorflow()
    
    def setup_tensorflow(self):
        """Configure TensorFlow for optimal performance"""
        # Enable mixed precision training if requested
        if self.config.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
        
        # Configure GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Using GPU: {gpus}")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}")
        else:
            logger.info("Using CPU for training")
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def load_training_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load training data from JSON with memory-efficient batch processing"""
        logger.info(f"Loading training data from {data_file}")
        
        # Check file size
        file_size = os.path.getsize(data_file) / (1024**3)  # GB
        logger.info(f"Data file size: {file_size:.2f} GB")
        
        if file_size > 1:  # If larger than 1GB, use batch loading
            return self._load_training_data_batch(data_file)
        else:
            return self._load_training_data_full(data_file)
    
    def _load_training_data_full(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load entire training data into memory"""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        input_audio = []
        output_audio = []
        control_params = []
        
        for sample in data:
            # Convert back to numpy arrays
            input_sig = np.array(sample['input_audio'])
            output_sig = np.array(sample['output_audio'])
            controls = sample['control_settings']
            
            # Create control parameter vector
            param_vector = self.extract_control_parameters(controls)
            
            # Segment into training sequences
            input_sequences, output_sequences = self.create_sequences(
                input_sig, output_sig, param_vector)
            
            input_audio.extend(input_sequences)
            output_audio.extend(output_sequences)
            control_params.extend([param_vector] * len(input_sequences))
        
        input_audio = np.array(input_audio)
        output_audio = np.array(output_audio)
        control_params = np.array(control_params)
        
        logger.info(f"Loaded {len(input_audio)} training sequences")
        return input_audio, output_audio, control_params
    
    def _load_training_data_batch(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load training data in batches to manage memory"""
        logger.info("Using batch loading for large dataset")
        
        # First pass: count total sequences
        total_sequences = 0
        with open(data_file, 'r') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            for sample in data:
                input_len = len(sample['input_audio'])
                num_seqs = (input_len - self.config.sequence_length) // (self.config.sequence_length // 2) + 1
                total_sequences += max(1, num_seqs)
        
        logger.info(f"Total sequences to generate: {total_sequences}")
        
        # Convert to HDF5 for efficient storage
        h5_file = data_file.replace('.json', '.h5')
        self._convert_to_hdf5(data_file, h5_file, total_sequences)
        
        # Load from HDF5
        with h5py.File(h5_file, 'r') as f:
            input_audio = f['input_audio'][:]
            output_audio = f['output_audio'][:]
            control_params = f['control_params'][:]
        
        return input_audio, output_audio, control_params
    
    def _convert_to_hdf5(self, json_file: str, h5_file: str, total_sequences: int):
        """Convert JSON data to HDF5 format for efficient loading"""
        if os.path.exists(h5_file):
            logger.info(f"HDF5 file already exists: {h5_file}")
            return
        
        logger.info(f"Converting {json_file} to HDF5 format...")
        
        with open(json_file, 'r') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        
        # Create HDF5 file
        with h5py.File(h5_file, 'w') as hf:
            # Pre-allocate arrays
            input_shape = (total_sequences, self.config.sequence_length)
            output_shape = (total_sequences, self.config.sequence_length)
            
            # Determine control parameter size
            first_controls = self.extract_control_parameters(data[0]['control_settings'])
            control_shape = (total_sequences, len(first_controls))
            
            input_dataset = hf.create_dataset('input_audio', input_shape, dtype='float32')
            output_dataset = hf.create_dataset('output_audio', output_shape, dtype='float32')
            control_dataset = hf.create_dataset('control_params', control_shape, dtype='float32')
            
            # Fill datasets
            idx = 0
            for sample in data:
                input_sig = np.array(sample['input_audio'], dtype=np.float32)
                output_sig = np.array(sample['output_audio'], dtype=np.float32)
                controls = self.extract_control_parameters(sample['control_settings'])
                
                # Create sequences
                input_seqs, output_seqs = self.create_sequences(input_sig, output_sig, controls)
                
                for i, (inp, out) in enumerate(zip(input_seqs, output_seqs)):
                    if idx < total_sequences:
                        input_dataset[idx] = inp[:, 0]  # Just audio, not controls
                        output_dataset[idx] = out
                        control_dataset[idx] = controls
                        idx += 1
        
        logger.info(f"HDF5 conversion complete: {h5_file}")
    
    def extract_control_parameters(self, controls: Dict) -> np.ndarray:
        """Extract control parameters into normalized vector"""
        # Define standard control parameters
        standard_params = [
            'gain', 'bass', 'mid', 'treble', 'presence', 'master',
            'input_level', 'bias_voltage', 'supply_voltage', 'load_impedance'
        ]
        
        param_vector = []
        
        # Extract standard parameters
        for param in standard_params:
            if param in controls:
                value = controls[param]
                # Normalize based on parameter type
                if param in ['gain', 'bass', 'mid', 'treble', 'presence', 'master']:
                    # 0-10 range controls
                    param_vector.append(value / 10.0)
                elif param == 'input_level':
                    # 0.001-1.0 range
                    param_vector.append(np.log10(value) / 3 + 1)  # Normalize to ~[0,1]
                elif param == 'bias_voltage':
                    # -2.0 to -0.5 range
                    param_vector.append((value + 2.0) / 1.5)
                elif param == 'supply_voltage':
                    # 280-360 range
                    param_vector.append((value - 280) / 80)
                elif param == 'load_impedance':
                    # 10k-1M range (log scale)
                    param_vector.append(np.log10(value / 10e3) / 2)
                else:
                    param_vector.append(0.0)
            else:
                param_vector.append(0.0)
        
        # Add channel encoding if available
        if 'channel' in controls or 'active_channel' in controls:
            channel = controls.get('channel', controls.get('active_channel', 1))
            # One-hot encode channel
            for i in range(1, 4):  # Support up to 3 channels
                param_vector.append(1.0 if channel == i else 0.0)
        
        # Add signal type encoding if available
        if 'signal_type' in controls:
            signal_types = ['sine', 'guitar_chord', 'white_noise', 'pink_noise', 'frequency_sweep', 'impulse']
            signal_encoding = [0.0] * len(signal_types)
            
            for st in signal_types:
                if st in controls['signal_type']:
                    idx = signal_types.index(st)
                    signal_encoding[idx] = 1.0
                    break
            
            param_vector.extend(signal_encoding)
        
        return np.array(param_vector, dtype=np.float32)
    
    def create_sequences(self, input_audio: np.ndarray, output_audio: np.ndarray, 
                        control_params: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Create overlapping sequences for training"""
        seq_length = self.config.sequence_length
        hop_length = seq_length // 2  # 50% overlap
        
        input_sequences = []
        output_sequences = []
        
        # Ensure both signals are same length
        min_length = min(len(input_audio), len(output_audio))
        input_audio = input_audio[:min_length]
        output_audio = output_audio[:min_length]
        
        # Create sequences
        for i in range(0, min_length - seq_length + 1, hop_length):
            input_seq = input_audio[i:i + seq_length]
            output_seq = output_audio[i:i + seq_length]
            
            # Combine audio with control parameters
            # Repeat control parameters for each sample in sequence
            control_seq = np.repeat(control_params[np.newaxis, :], 
                                  seq_length, axis=0)
            
            # Concatenate input audio with control parameters
            input_with_controls = np.column_stack([
                input_seq.reshape(-1, 1), 
                control_seq
            ])
            
            input_sequences.append(input_with_controls)
            output_sequences.append(output_seq)
        
        return input_sequences, output_sequences
    
    def create_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create neural network model"""
        logger.info(f"Creating {self.config.model_type} model with input shape {input_shape}")
        
        if self.config.model_type == "lstm":
            return self.create_lstm_model(input_shape)
        elif self.config.model_type == "cnn":
            return self.create_cnn_model(input_shape)
        elif self.config.model_type == "wavenet":
            return self.create_wavenet_model(input_shape)
        elif self.config.model_type == "transformer":
            return self.create_transformer_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create LSTM-based model with regularization"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Normalization
            layers.LayerNormalization(),
            
            # LSTM layers with residual connections
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LayerNormalization(),
            layers.Dropout(0.2),
            
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LayerNormalization(),
            layers.Dropout(0.2),
            
            layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LayerNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dense(1, activation='tanh')  # Output audio
        ])
        
        return model
    
    def create_cnn_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create CNN-based model (WaveNet style) with skip connections"""
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv1D(64, 3, padding='causal', activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        
        skip_connections = []
        
        # Dilated convolution blocks
        for dilation_rate in [1, 2, 4, 8, 16, 32, 64, 128]:
            # Residual block start
            residual = x
            
            # Dilated convolution
            conv = layers.Conv1D(64, 3, dilation_rate=dilation_rate, 
                               padding='causal', activation=None)(x)
            conv = layers.LayerNormalization()(conv)
            
            # Gated activation
            filter_conv = layers.Conv1D(64, 1, activation='tanh')(conv)
            gate_conv = layers.Conv1D(64, 1, activation='sigmoid')(conv)
            gated = layers.Multiply()([filter_conv, gate_conv])
            
            # Skip connection
            skip = layers.Conv1D(64, 1)(gated)
            skip_connections.append(skip)
            
            # Residual connection
            residual_out = layers.Conv1D(64, 1)(gated)
            x = layers.Add()([residual, residual_out])
        
        # Combine skip connections
        skip_sum = layers.Add()(skip_connections) if len(skip_connections) > 1 else skip_connections[0]
        
        # Output layers
        x = layers.Conv1D(32, 1, activation='relu')(skip_sum)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Conv1D(1, 1, activation='tanh')(x)
        
        return keras.Model(inputs, outputs)
    
    def create_wavenet_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create full WaveNet-style model"""
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv1D(32, 2, padding='causal')(inputs)
        x = layers.LayerNormalization()(x)
        
        # Dilated convolution stack
        skip_connections = []
        
        for stack in range(2):  # 2 stacks for efficiency
            for dilation_rate in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                # Store input for residual
                res_input = x
                
                # Dilated convolution
                conv = layers.Conv1D(32, 2, dilation_rate=dilation_rate, 
                                   padding='causal', activation=None)(x)
                conv = layers.LayerNormalization()(conv)
                
                # Gated activation unit
                filter_conv = layers.Conv1D(32, 1, activation='tanh')(conv)
                gate_conv = layers.Conv1D(32, 1, activation='sigmoid')(conv)
                gated = layers.Multiply()([filter_conv, gate_conv])
                
                # Skip connection
                skip = layers.Conv1D(32, 1)(gated)
                skip_connections.append(skip)
                
                # Residual connection
                residual = layers.Conv1D(32, 1)(gated)
                x = layers.Add()([res_input, residual])
        
        # Combine skip connections
        skip_sum = layers.Add()(skip_connections)
        
        # Output layers
        x = layers.Conv1D(64, 1, activation='relu')(skip_sum)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv1D(32, 1, activation='relu')(x)
        outputs = layers.Conv1D(1, 1, activation='tanh')(x)
        
        return keras.Model(inputs, outputs)
    
    def create_transformer_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create Transformer-based model for sequence modeling"""
        inputs = layers.Input(shape=input_shape)
        
        # Initial projection
        x = layers.Dense(64)(inputs)
        x = layers.LayerNormalization()(x)
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = layers.Embedding(input_shape[0], 64)(positions)
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(4):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=4, key_dim=64, dropout=0.2
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward network
            ffn = keras.Sequential([
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64)
            ])(x)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)
        
        # Output projection
        outputs = layers.Dense(1, activation='tanh')(x)
        
        return keras.Model(inputs, outputs)
    
    def spectral_loss(self, y_true, y_pred):
        """Multi-scale spectral loss function"""
        loss = 0.0
        
        # Multiple FFT window sizes
        fft_sizes = [64, 128, 256, 512, 1024, 2048]
        
        for fft_size in fft_sizes:
            if fft_size > tf.shape(y_true)[1]:
                continue
            
            # Compute STFT
            true_stft = tf.signal.stft(
                tf.squeeze(y_true, axis=-1),
                frame_length=fft_size,
                frame_step=fft_size//4,
                fft_length=fft_size
            )
            
            pred_stft = tf.signal.stft(
                tf.squeeze(y_pred, axis=-1),
                frame_length=fft_size,
                frame_step=fft_size//4,
                fft_length=fft_size
            )
            
            # Magnitude loss
            true_mag = tf.abs(true_stft)
            pred_mag = tf.abs(pred_stft)
            mag_loss = tf.reduce_mean(tf.abs(true_mag - pred_mag))
            
            # Log magnitude loss for better perceptual weighting
            true_log_mag = tf.math.log(true_mag + 1e-7)
            pred_log_mag = tf.math.log(pred_mag + 1e-7)
            log_mag_loss = tf.reduce_mean(tf.abs(true_log_mag - pred_log_mag))
            
            # Phase loss (wrapped)
            true_phase = tf.math.angle(true_stft)
            pred_phase = tf.math.angle(pred_stft)
            phase_diff = tf.math.atan2(
                tf.sin(true_phase - pred_phase),
                tf.cos(true_phase - pred_phase)
            )
            phase_loss = tf.reduce_mean(tf.abs(phase_diff))
            
            # Combine losses for this scale
            scale_loss = mag_loss + 0.5 * log_mag_loss + 0.1 * phase_loss
            loss += scale_loss / len(fft_sizes)
        
        return loss
    
    def combined_loss(self, y_true, y_pred):
        """Combined time-domain and spectral loss"""
        # Time-domain loss (MSE)
        time_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Spectral loss
        spectral_loss = self.spectral_loss(y_true, y_pred)
        
        # Combined loss
        total_loss = (self.config.time_loss_weight * time_loss + 
                     self.config.spectral_loss_weight * spectral_loss)
        
        return total_loss
    
    def augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation techniques"""
        if not self.config.use_data_augmentation:
            return X, y
        
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            # Original data
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # Random gain variation
            gain = np.random.uniform(0.8, 1.2)
            augmented_X.append(X[i] * gain)
            augmented_y.append(y[i] * gain)
            
            # Add slight noise
            noise_level = self.config.augmentation_factor
            noise_X = X[i] + np.random.normal(0, noise_level, X[i].shape)
            noise_y = y[i] + np.random.normal(0, noise_level * 0.5, y[i].shape)
            augmented_X.append(noise_X)
            augmented_y.append(noise_y)
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def train_model(self, input_audio: np.ndarray, output_audio: np.ndarray, 
                   control_params: np.ndarray,
                   use_generator: bool = False) -> keras.Model:
        """Train the neural network model with improved training loop"""
        logger.info("Starting model training...")
        
        # Split data
        indices = np.arange(len(input_audio))
        train_idx, temp_idx = train_test_split(
            indices, test_size=self.config.validation_split + self.config.test_split,
            random_state=42
        )
        
        val_size = self.config.validation_split / (self.config.validation_split + self.config.test_split)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=1-val_size, random_state=42
        )
        
        if use_generator:
            # Use data generators for large datasets
            train_gen = DataGenerator(
                'training_data.h5', self.config.batch_size,
                self.config.sequence_length, train_idx
            )
            val_gen = DataGenerator(
                'training_data.h5', self.config.batch_size,
                self.config.sequence_length, val_idx, shuffle=False
            )
        else:
            # Prepare data in memory
            X_train = input_audio[train_idx]
            X_val = input_audio[val_idx]
            y_train = output_audio[train_idx]
            y_val = output_audio[val_idx]
            
            # Apply data augmentation
            X_train, y_train = self.augment_data(X_train, y_train)
            
            # Reshape for model input
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
            y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        
        logger.info(f"Training data shape: {X_train.shape if not use_generator else 'Using generator'}")
        logger.info(f"Validation data shape: {X_val.shape if not use_generator else 'Using generator'}")
        
        # Create model
        input_shape = X_train.shape[1:] if not use_generator else (self.config.sequence_length, input_audio.shape[-1])
        self.model = self.create_model(input_shape)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        if self.config.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        if self.config.use_spectral_loss:
            loss_fn = self.combined_loss
        else:
            loss_fn = 'mse'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae', 'mse']
        )
        
        # Print model summary
        self.model.summary()
        
        # Save initial weights for potential retraining
        initial_weights = self.model.get_weights()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                self.checkpoint_dir / 'best_model.h5',
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            keras.callbacks.CSVLogger('training_log.csv')
        ]
        
        # Custom callback for additional monitoring
        class TrainingMonitor(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    logger.info(f"Epoch {epoch + 1}: loss={logs['loss']:.4f}, "
                              f"val_loss={logs.get('val_loss', 'N/A')}")
        
        callbacks.append(TrainingMonitor())
        
        # Train model
        try:
            if use_generator:
                self.training_history = self.model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=self.config.epochs,
                    callbacks=callbacks,
                    verbose=1,
                    workers=4,
                    use_multiprocessing=True
                )
            else:
                self.training_history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        logger.info("Model training completed")
        
        # Evaluate on test set
        if not use_generator:
            X_test = input_audio[test_idx]
            y_test = output_audio[test_idx]
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
            y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
            
            test_metrics = self.evaluate_model(X_test, y_test)
            logger.info(f"Test set performance: {test_metrics}")
        
        return self.model
    
    def evaluate_model(self, test_input: np.ndarray, test_output: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics"""
        logger.info("Evaluating model performance...")
        
        # Make predictions
        predictions = self.model.predict(test_input, batch_size=self.config.batch_size)
        
        # Calculate metrics
        mse = np.mean((test_output - predictions) ** 2)
        mae = np.mean(np.abs(test_output - predictions))
        
        # Signal-to-noise ratio
        signal_power = np.mean(test_output ** 2)
        noise_power = np.mean((test_output - predictions) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Frequency response correlation
        freq_corr = self.calculate_frequency_response_correlation(
            test_output, predictions)
        
        # PESQ-like perceptual metric (simplified)
        perceptual_score = self.calculate_perceptual_metric(
            test_output, predictions)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'snr_db': float(snr),
            'frequency_correlation': float(freq_corr),
            'perceptual_score': float(perceptual_score)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def calculate_frequency_response_correlation(self, y_true: np.ndarray, 
                                              y_pred: np.ndarray) -> float:
        """Calculate frequency response correlation"""
        # Average frequency response
        true_fft = np.mean([np.abs(np.fft.fft(y[..., 0])) for y in y_true], axis=0)
        pred_fft = np.mean([np.abs(np.fft.fft(y[..., 0])) for y in y_pred], axis=0)
        
        # Correlation coefficient
        correlation = np.corrcoef(true_fft[:len(true_fft)//2], 
                                pred_fft[:len(pred_fft)//2])[0, 1]
        return correlation
    
    def calculate_perceptual_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate a simplified perceptual quality metric"""
        # This is a simplified version - for production, use PESQ or similar
        
        # Spectral centroid difference
        def spectral_centroid(signal):
            magnitude = np.abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1/self.config.sample_rate)
            return np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        
        true_centroids = [spectral_centroid(y[..., 0]) for y in y_true[:100]]  # Sample
        pred_centroids = [spectral_centroid(y[..., 0]) for y in y_pred[:100]]
        
        centroid_error = np.mean(np.abs(np.array(true_centroids) - np.array(pred_centroids)))
        
        # Convert to perceptual score (0-5 scale, 5 being best)
        score = 5.0 * np.exp(-centroid_error / 1000)
        return np.clip(score, 0, 5)
    
    def save_model(self, model_name: str = "neural_amp_model"):
        """Save trained model and metadata"""
        logger.info(f"Saving model as {model_name}")
        
        # Create model directory
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model in multiple formats
        # 1. Keras native format
        model_path = model_dir / f"{model_name}.h5"
        self.model.save(model_path)
        
        # 2. SavedModel format
        savedmodel_path = model_dir / "saved_model"
        self.model.save(savedmodel_path, save_format='tf')
        
        # Save configuration
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save training history
        if self.training_history:
            history_path = model_dir / "training_history.json"
            with open(history_path, 'w') as f:
                history_dict = {k: [float(x) for x in v] 
                              for k, v in self.training_history.history.items()}
                json.dump(history_dict, f, indent=2)
        
        # Save model architecture
        architecture_path = model_dir / "architecture.json"
        with open(architecture_path, 'w') as f:
            f.write(self.model.to_json())
        
        # Save scalers if used
        scaler_path = model_dir / "scalers.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'input_scaler': self.scaler_input,
                'output_scaler': self.scaler_output
            }, f)
        
        logger.info(f"Model saved to {model_dir}")
    
    def plot_training_history(self):
        """Plot training history with improved visualization"""
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        history = self.training_history.history
        epochs = range(1, len(history['loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_yscale('log')
        
        # MAE plot
        axes[0, 1].plot(epochs, history['mae'], 'b-', label='Training MAE')
        axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot (if available)
        if 'lr' in history:
            axes[1, 0].plot(epochs, history['lr'], 'g-')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
            axes[1, 0].set_yscale('log')
        
        # MSE plot
        if 'mse' in history:
            axes[1, 1].plot(epochs, history['mse'], 'b-', label='Training MSE')
            axes[1, 1].plot(epochs, history['val_mse'], 'r-', label='Validation MSE')
            axes[1, 1].set_title('Mean Squared Error')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300)
        plt.show()
    
    def export_for_realtime(self, model_name: str = "neural_amp_model"):
        """Export model for real-time use in various formats"""
        logger.info("Exporting model for real-time deployment...")
        
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # 1. TensorFlow Lite conversion
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Representative dataset for quantization
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, self.config.sequence_length, 
                                         self.model.input_shape[-1]).astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_dataset
            
            # Convert to TFLite
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = model_dir / f"{model_name}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite model saved to {tflite_path}")
            
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
        
        # 2. ONNX conversion
        try:
            import tf2onnx
            
            onnx_path = model_dir / f"{model_name}.onnx"
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                output_path=str(onnx_path),
                opset=13
            )
            logger.info(f"ONNX model saved to {onnx_path}")
            
        except ImportError:
            logger.warning("tf2onnx not installed, skipping ONNX export")
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
        
        # 3. C++ header file generation for embedded systems
        self._generate_cpp_header(model_dir / f"{model_name}.h")
        
        # 4. VST plugin template
        self._generate_vst_template(model_dir / "vst_plugin")
        
        logger.info(f"Real-time models exported to {model_dir}")
    
    def _generate_cpp_header(self, output_path: Path):
        """Generate C++ header file with model weights"""
        logger.info("Generating C++ header file...")
        
        with open(output_path, 'w') as f:
            f.write("// Auto-generated neural amp model\n")
            f.write(f"// Model: {self.config.model_type}\n")
            f.write(f"// Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("#ifndef NEURAL_AMP_MODEL_H\n")
            f.write("#define NEURAL_AMP_MODEL_H\n\n")
            
            f.write("#include <vector>\n")
            f.write("#include <array>\n\n")
            
            f.write("namespace NeuralAmp {\n\n")
            
            # Model metadata
            f.write(f"constexpr int SEQUENCE_LENGTH = {self.config.sequence_length};\n")
            f.write(f"constexpr int SAMPLE_RATE = {self.config.sample_rate};\n\n")
            
            # Model weights (simplified - in practice, serialize all weights)
            f.write("// Model weights would be serialized here\n")
            f.write("// This is a template for the actual implementation\n\n")
            
            f.write("class NeuralAmpModel {\n")
            f.write("public:\n")
            f.write("    NeuralAmpModel();\n")
            f.write("    float process(float input, const std::vector<float>& controls);\n")
            f.write("private:\n")
            f.write("    // Model state and weights\n")
            f.write("};\n\n")
            
            f.write("} // namespace NeuralAmp\n\n")
            f.write("#endif // NEURAL_AMP_MODEL_H\n")
        
        logger.info(f"C++ header generated: {output_path}")
    
    def _generate_vst_template(self, output_dir: Path):
        """Generate VST plugin template"""
        output_dir.mkdir(exist_ok=True)
        
        # Generate CMakeLists.txt
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write("cmake_minimum_required(VERSION 3.15)\n")
            f.write("project(NeuralAmpVST)\n\n")
            f.write("# VST3 SDK path\n")
            f.write("set(VST3_SDK_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/vst3sdk\")\n\n")
            f.write("# Add VST3 plugin target\n")
            f.write("add_library(NeuralAmpVST MODULE\n")
            f.write("    src/PluginProcessor.cpp\n")
            f.write("    src/PluginEditor.cpp\n")
            f.write("    src/NeuralAmpModel.cpp\n")
            f.write(")\n\n")
            f.write("# Link libraries\n")
            f.write("target_link_libraries(NeuralAmpVST PRIVATE\n")
            f.write("    ${VST3_SDK_PATH}/public.sdk\n")
            f.write(")\n")
        
        logger.info(f"VST plugin template generated: {output_dir}")


def main():
    """Example usage of Neural Amp Trainer"""
    
    # Configuration
    config = TrainingConfig(
        model_type="lstm",
        sequence_length=512,
        batch_size=32,
        epochs=100,
        learning_rate=0.001,
        use_spectral_loss=True,
        use_mixed_precision=True,
        use_data_augmentation=True
    )
    
    # Initialize trainer
    trainer = NeuralAmpTrainer(config)
    
    try:
        # Load training data
        logger.info("Loading training data...")
        input_audio, output_audio, control_params = trainer.load_training_data(
            "tube_preamp_training_data.json"
        )
        
        # Train model
        logger.info("Training neural network...")
        model = trainer.train_model(
            input_audio, output_audio, control_params,
            use_generator=len(input_audio) > 10000  # Use generator for large datasets
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        test_size = min(1000, len(input_audio) // 10)
        metrics = trainer.evaluate_model(
            input_audio[:test_size], 
            output_audio[:test_size]
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        # Save model
        trainer.save_model("tube_preamp_model_v1")
        
        # Export for real-time use
        trainer.export_for_realtime("tube_preamp_model_v1")
        
        logger.info("Neural amp model training completed successfully!")
        logger.info(f"Final metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
