# Change Point Detection with Computer Vision

If you think manually sifting through signals to spot changes isn't so funny, this is for you! This project uses the magic of computer vision to automate the detection of sudden increases or decreases in a signal.

##### But why computer vision?

How do you accurately define a "sudden increase" in a signal? A big difference between a point and the next one? A big change in the derivative? But how big? It depends. And how do you treat noise and the various edge cases?

Meanwhile, when you simply look at the plot you immediately see it. Let's teach the computer how to do it!

---

### Features

* **YOLOv8n-based Detection**: A robust object detection model trained to find "increase" and "decrease" events in signal plots.

* **Custom YOLO-like Model**: A lightweight, from-scratch model built with PyTorch, specifically for this task.

* **Visual Approach**: The entire method is based on turning signals into images and then using classic computer vision techniques.

---

### How it Works

The project transforms time-series data into images and then uses computer vision models to identify key events.

1.  **Data Generation**: Signal data from lab sensors are plotted using `matplotlib` to create an image dataset.

2.  **Annotation**: These plots are then annotated with bounding boxes using `labelImg` to mark where the increases and decreases occur.

3.  **Model Training**:

    * **YOLOv8**: A YOLOv8n model is trained on the annotated dataset.

    * **From-Scratch Model**: A custom model is built from the ground up, designed to work with simplified, binarized grayscale images. It classifies each grid cell as "increase," "decrease," or "background."

---

### Results

The models were tested on unseen data and showed promising results! Here are a couple of examples:

#### YOLOv8 Test Results

![YOLOv8 Test Results](https://github.com/davide710/change_detection/blob/main/inference_output/149.jpg)

Other test results [here](https://github.com/davide710/change_detection/tree/main/inference_output)

#### Custom Model Test Results

At the moment, the from-scratch model outputs three (80x80, 40x40, 20x20) grids, and for each cell it predicts whether there is an increase, a decrease or nothing.
To show the results, cells classified as "increase" were drawn with solid edges, cells classified as "decrease" with dotted edges.

![Custom Model Test Results](https://github.com/davide710/change_detection/blob/main/from_scratch/predictions_v1/149.png)

Other test results [here](https://github.com/davide710/change_detection/tree/main/from_scratch/predictions_v1)

---

### Usage

-soon

---

### Contributions Welcome!

This project is still a work in progress, and there's plenty of room for improvement. If you find it interesting, you can contribute!

Here are some ideas for future work:

* **Improve the from-scratch model**: The custom model's loss function, architecture, and training can be improved for better performance.

* **Create a user-friendly script**: A script that takes a numpy array as input, processes it, and returns the change detection results.

* **Expand the dataset**: More diverse signal data would help the models generalize even better.

Feel free to fork the repository, make changes, and submit a pull request!
