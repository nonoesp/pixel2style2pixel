from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.serialization import load_lua
import os
import cv2
import numpy as np
import argparse

"""
NOTE!: Must have torch==0.4.1 and torchvision==0.2.1
The sketch simplification model (sketch_gan.t7) from Simo Serra et al. can be downloaded from their official implementation: 
    https://github.com/bobbens/sketch_simplification
"""


def sobel(img, ksize=3):
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=ksize)
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=ksize)
    return cv2.bitwise_or(opImgx, opImgy)


def sketch(frame, blur_size=3, sobel_size=3):
    frame = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    invImg = 255 - frame
    edgImg0 = sobel(frame, sobel_size)
    edgImg1 = sobel(invImg, sobel_size)
    edgImg = cv2.addWeighted(edgImg0, 0.75, edgImg1, 0.75, 0)
    opImg = 255 - edgImg
    return opImg


def get_sketch_image(image_path, blur_size, sobel_size):
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    sketch_image = sketch(original, blur_size, sobel_size)
    return sketch_image[:, :, np.newaxis]


def simplify(input_dir, output_dir, model, blur_size=3, sobel_size=3):

    use_cuda = True

    cache = load_lua(model)
    model = cache.model
    immean = cache.mean
    imstd = cache.std
    model.evaluate()

    images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, image_path in enumerate(images):
        if idx % 50 == 0:
            print("{} out of {}".format(idx, len(images)))
        data = get_sketch_image(image_path, blur_size, sobel_size)
        data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
        if use_cuda:
            pred = model.cuda().forward(data.cuda()).float()
        else:
            pred = model.forward(data)
        save_image(pred[0], os.path.join(output_dir, "{}_edges.jpg".format(
            image_path.split("/")[-1].split('.')[0])))

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate sketch data from images.')
    parser.add_argument('-i', '--input', required=True, help='Directory containing input images.')
    parser.add_argument('-o', '--output', required=True, help='Directory containing output sketches.')
    parser.add_argument('-m', '--model', required=True, help='Path to pre-trained model.')
    parser.add_argument('-b', '--blur', default=3, type=float, help='Gaussian blur kernel size.')
    parser.add_argument('-s', '--sobel', default=3, type=float, help='Sobel kernel size.')
    opt = parser.parse_args()
    
    simplify(opt.input, opt.output, opt.model, opt.blur, opt.sobel)