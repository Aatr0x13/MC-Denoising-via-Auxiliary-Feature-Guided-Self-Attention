import os
from preprocessing import *
import matplotlib
import matplotlib.pyplot as plt
import pyexr
matplotlib.use('Agg')


# =========================================================exr==========================================================
def _show_data(data, channel):
    figsize = (15, 15)
    plt.figure(figsize=figsize)
    plt.title(channel)
    img_plot = plt.imshow(data, aspect='equal')
    img_plot.axes.get_xaxis().set_visible(False)
    img_plot.axes.get_yaxis().set_visible(False)
    plt.show()


def process_data(data, channel, width, height):
    if channel in ["default", "target", "diffuse", "albedo", "specular"]:
        data = np.clip(data, 0, 1) ** 0.45454545
    elif channel in ["normal", "normalA"]:
        # normalize
        for i in range(height):
            for j in range(width):
                data[i][j] = data[i][j] / np.linalg.norm(data[i][j])
        data = np.abs(data)
    elif channel in ["depth", "visibility", "normalVariance"] and np.max(data) != 0:
        data /= np.max(data)

    if data.shape[2] == 1:
        # reshape
        data = data.reshape(height, width)

    return data


def show_exr_info(exr_path):
    assert exr_path, 'Exr_path cannot be empty.'
    assert exr_path.endswith('exr'), "Img to be shown must be in '.exr' format."
    exr = pyexr.open(exr_path)
    print("Width:", exr.width)
    print("Height:", exr.height)
    print("Available channels:")
    exr.describe_channels()
    print("Default channels:", exr.channel_map['default'])


def show_exr_channel(exr_path, channel):
    exr = pyexr.open(exr_path)
    data = exr.get(channel)
    print("Channel:", channel)
    print("Shape:", data.shape)
    print("Max: %f    Min: %f" % (np.max(data), np.min(data)))
    data = process_data(data, channel, exr.width, exr.height)
    _show_data(data, channel)


# ========================================================img===========================================================
def tone_mapping(matrix, gamma=2.2):
    return np.clip(matrix ** (1.0 / gamma), 0, 1)


def tensor2img(image_numpy, post_spec=False, post_diff=False, albedo=None):
    if post_diff:
        assert albedo is not None, "must provide albedo when post_diff is True"
    image_type = np.uint8

    # multiple images
    if image_numpy.ndim == 4:
        temp = []
        for i in range(len(image_numpy)):
            if post_diff:
                temp.append(tensor2img(image_numpy[i], post_spec=False, post_diff=True, albedo=albedo[i]))
            else:
                temp.append(tensor2img(image_numpy[i], post_spec=post_spec, post_diff=False))
        return np.array(temp)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    # postprocessing
    if post_spec:
        image_numpy = postprocess_specular(image_numpy)
    elif post_diff:
        albedo = np.transpose(albedo, (1, 2, 0))
        image_numpy = postprocess_diffuse(image_numpy, albedo)
    image_numpy = tone_mapping(image_numpy) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype(image_type)

    return image_numpy


def save_img(save_path, img, figsize, dpi, color=None):
    plt.cla()
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis("off")
    plt.imshow(img)
    fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    if color:
        plt.gca().add_patch(
            plt.Rectangle(xy=(0, 0), width=img.shape[1], height=img.shape[0], edgecolor=color, fill=False,
                          linewidth=img.shape[1]*1/92))
    fig.savefig(save_path, format='png', transparent=True, pad_inches=0)


def save_img_group(save_path, iter, noisy, output, y):
    name = os.path.join(save_path, "%d.png" % iter)
    # multiple images, just save the first one
    if noisy.ndim == 4:
        noisy = noisy[0]
        output = output[0]
        y = y[0]
    plt.subplot(131)
    plt.axis("off")
    plt.imshow(noisy)
    plt.title("Noisy")

    plt.subplot(132)
    plt.axis("off")
    plt.imshow(output)
    plt.title("Output")

    plt.subplot(133)
    plt.axis("off")
    plt.imshow(y)
    plt.title("Reference")
    plt.savefig(name, bbox_inches='tight')


# ========================================================util==========================================================
def create_folder(path, still_create=False):
    '''
    :param still_create: still create or not when there's already a folder with the same name
    :return: path to the created folder
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    elif still_create:
        if '\\' in path:
            dir_root = path[: path.rfind('\\')]
        else:
            dir_root = '.'
        count = 1
        original_dir_name = path.split('\\')[-1]
        while True:
            dir_name = original_dir_name + '_%d' % count
            path = os.path.join(dir_root, dir_name)
            if os.path.exists(path):
                count += 1
            else:
                os.mkdir(path)
                break
    return path

