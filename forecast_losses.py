"""
Custom losses for multi-step power forecasting.

Plain MSE often encourages smooth mean-like forecasts for noisy multi-step
targets. ForecastShapeLoss keeps the level error while also matching local
changes and output variability.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="PowerForecast")
class ForecastShapeLoss(tf.keras.losses.Loss):
    def __init__(self, level_weight=1.0, diff_weight=1.5, std_weight=1.0,
                 name="forecast_shape_loss", reduction="sum_over_batch_size",
                 **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.level_weight = level_weight
        self.diff_weight = diff_weight
        self.std_weight = std_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        level_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

        n_steps = tf.shape(y_true)[-1]

        def compute_diff_loss():
            true_diff = y_true[..., 1:] - y_true[..., :-1]
            pred_diff = y_pred[..., 1:] - y_pred[..., :-1]
            return tf.reduce_mean(tf.square(true_diff - pred_diff), axis=-1)

        diff_loss = tf.cond(
            n_steps > 1,
            compute_diff_loss,
            lambda: tf.zeros_like(level_loss)
        )

        true_centered = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
        pred_centered = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        true_std = tf.sqrt(tf.reduce_mean(tf.square(true_centered), axis=-1) + 1e-6)
        pred_std = tf.sqrt(tf.reduce_mean(tf.square(pred_centered), axis=-1) + 1e-6)
        std_loss = tf.square(true_std - pred_std)

        return (
            self.level_weight * level_loss +
            self.diff_weight * diff_loss +
            self.std_weight * std_loss
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "level_weight": self.level_weight,
            "diff_weight": self.diff_weight,
            "std_weight": self.std_weight,
        })
        return config


def make_forecast_loss():
    return ForecastShapeLoss(level_weight=1.0, diff_weight=1.5, std_weight=1.0)
