import numpy as np 
import gradio as gr 
import scipy as sp
from skimage import color


def serpia(img):
    serpia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    serpia_img = img.dot(serpia_filter.T)
    serpia_img = serpia_img/serpia_img.max()

    return serpia_img

def sobel(img):
    img = color.rgb2gray(img)
    sx = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])
    sy = np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ])

    gx = sp.ndimage.convolve(img, sx)
    gy = sp.ndimage.convolve(img, sy)

    edges = np.sqrt(gx**2+gy**2)
    edges /= edges.max()

    return edges


demo = gr.Interface(sobel, gr.Image(), "image")

if __name__ == "__main__":
    demo.launch()