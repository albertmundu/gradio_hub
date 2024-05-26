import numpy as np 
import gradio as gr 
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt
from skimage import color, filters, io


def serpia(img):
    serpia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    serpia_img = img.dot(serpia_filter.T)
    serpia_img = serpia_img/serpia_img.max()

    return serpia_img

def apply_filter(image, filter_type):
    #convert the image to grayscale

    if filter_type in ["Sobel", "Scharr", "Prewitt"]:
        gray_image = color.rgb2gray(image)

        if filter_type == "Sobel":
            edge_magnitude = filters.sobel(gray_image)
        
        elif filter_type == "Scharr":
            edge_magnitude = filters.scharr(gray_image)
        elif filter_type == "Prewitt":
            edge_magnitude = filters.prewitt(gray_image)

        edge_magnitude = edge_magnitude/np.max(edge_magnitude)

        return edge_magnitude
    elif filter_type == "Serpia":
        return serpia(image)
    elif filter_type == "Gaussian":
        return gaussian_filter(image, sigma=1)

def plot_histogram(image):
    fig, ax = plt.subplots()
    ax.hist(image.ravel(), bins=256, range=[0,1], color="blue")
    ax.set_title('Pixel Intensity Histogram')
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Frequency')
    plt.close(fig)
    return fig


def process_image(image, filter_type):
    img = apply_filter(image, filter_type)
    gray_image = color.rgb2gray(image)
    input_hist = plot_histogram(gray_image)
    output_hist = plot_histogram(img)
    return img, input_hist, output_hist


if __name__ == "__main__":

    # inputs = [
    #     gr.Image(type="numpy", label="Input Image"),
    #     gr.Radio(choices=["Sobel","Scharr","Prewitt", "Serpia"], label="Filter Type", value="Serpia")
    # ]

    # outputs = [
    #     gr.Image(type="numpy", label="Filtered Image"),
    #     gr.Plot(label="Input Histogram"),
    #     gr.Plot(label="Output Histogram"),
    # ]

    # gr.Interface(fn=process_image, inputs=inputs, outputs=outputs, title="Apply Image Filters").launch()


    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("Edge Detection"):
                with gr.Row():
                    with gr.Column(min_width=200, scale=1):
                        input_image1 = gr.Image(type="numpy", label="Input")
                        edge_btn = gr.Button("Detect Edges")
                    radio_btn1 = gr.Radio(choices=["Sobel","Scharr","Prewitt"], label="Filter Type", value="Sobel")
                
                with gr.Row():
                    output_image1 = gr.Image(type="numpy", label="Output")
                    hist_input1 = gr.Plot(label="Input Histogram")
                    hist_output1 = gr.Plot(label="Output Histogram")
            with gr.Tab("Apply Filters"):
                with gr.Row():
                    with gr.Column(min_width=200, scale=1):
                        input_image2 = gr.Image(type="numpy", label="Input")
                        filter_btn = gr.Button("Apply Filter")
                    radio_btn2 = gr.Radio(choices=["Serpia", "Gaussian"], label="Filter Type", value="Serpia") 
                with gr.Row():
                    output_image2 = gr.Image(type="numpy", label="Output")
                    hist_input2 = gr.Plot(label="Input Histogram")
                    hist_output2 = gr.Plot(label="Output Histogram")  


        edge_btn.click(fn=process_image, inputs=[input_image1, radio_btn1], outputs=[output_image1, hist_input1, hist_output1])
        filter_btn.click(fn=process_image, inputs=[input_image2, radio_btn2], outputs=[output_image2, hist_input2, hist_output2])

    demo.launch()