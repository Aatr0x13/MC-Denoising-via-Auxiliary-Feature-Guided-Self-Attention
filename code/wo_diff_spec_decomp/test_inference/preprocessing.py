import numpy as np


# constants
eps = 0.00316


def preprocess_diffuse(diffuse, albedo):
    return diffuse / (albedo + eps)


def preprocess_specular(specular):
    # assert np.sum(specular < 0) == 0, "Negative value in specular component!"
    return np.log(specular + 1)


def preprocess_depth(depth):
    depth = np.clip(depth, 0.0, np.max(depth))
    max_feature = np.max(depth)
    if max_feature != 0:
        depth /= max_feature
    return depth


def preprocess_normal(normal):
    normal = np.nan_to_num(normal)
    normal = (normal + 1.0) * 0.5
    normal = np.maximum(np.minimum(normal, 1.0), 0.0)
    return normal


def postprocess_diffuse(diffuse, albedo):
    return diffuse * (albedo + eps)


def postprocess_specular(specular):
    return np.exp(specular) - 1




