from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import math


def train_model(model, train_generator, validation_generator, num_train_samples, num_val_samples, batch_size, epochs):
    try:
        model.compile(optimizer=Adam(learning_rate=1e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_model_efficientnet_b3.keras', save_best_only=True, monitor='val_loss', verbose=1)
        ]

        steps_per_epoch = math.ceil(num_train_samples / batch_size)
        validation_steps = math.ceil(num_val_samples / batch_size)

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        model.save('model_efficientnet_b3.keras')

        return history
    except Exception as e:
        print(f"Váratlan hiba történt a modell tanítása során: {str(e)}")
        return None
