import numpy as np


def sine_wave(x):
    return np.sin(x)


def square_wave(x):
    return np.sign(np.sin(x))


def chirp_signal(x):
    return np.sin(0.5 * x**2)


def gaussian_pulse(x):
    return np.exp(-0.5 * ((x - 2 * np.pi) / 0.5) ** 2)


def white_noise(x):
    return np.random.normal(0, 0.5, x.shape)


def step_function(x):
    return np.heaviside(x - 2 * np.pi, 1)


def sawtooth_wave(x):
    return 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))


def triangle_wave(x):
    return 2 * np.abs(2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))) - 1


def exponential_decay(x):
    return np.exp(-x / (2 * np.pi))


def rectified_sine_wave(x):
    return np.abs(np.sin(x))


def am_signal(x):
    return (1 + 0.5 * np.sin(x)) * np.sin(5 * x)


def fm_signal(x):
    return np.sin(5 * x + np.sin(x))


SIGNALS_2D = {
    "sine_wave": sine_wave,
    "square_wave": square_wave,
    "chirp_signal": chirp_signal,
    "gaussian_pulse": gaussian_pulse,
    "white_noise": white_noise,
    "step_function": step_function,
    "sawtooth_wave": sawtooth_wave,
    "triangle_wave": triangle_wave,
    "exponential_decay": exponential_decay,
    "rectified_sine_wave": rectified_sine_wave,
    "am_signal": am_signal,
    "fm_signal": fm_signal,
}
