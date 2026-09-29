---
layout: post
title: "An ASL and Emotion Overlay for Any Video Call"
tab_title: SignifyAI
code: https://github.com/SadeekFarhan21/projects/tree/main/signifyai
date: 2025-02-22 10:35:49
tags:
  - computer-vision
  - tensorflow
  - sign-language
description: >-
  SignifyAI is a three-day team build that labels ASL letters, hand gestures and
  facial emotion over any window, including a video call, with three small CNNs
  that score 99.8% on gestures, 68.2% on FER2013 emotion and about 98% on ASL.
---

Most sign-language demos own the webcam. We wanted something that helps deaf and hearing people talk on the video calls they already use, so SignifyAI does the opposite. It is a transparent, always-on-top overlay that sits over whatever is already on screen and labels the hands and faces in it with an ASL letter, a hand gesture and an emotion. Jalen Francis, Jayson Clark and I built it at Ohio State over three days in February 2025, in Python with TensorFlow and PyQt5, on public Kaggle datasets: an ASL alphabet set, LeapGestRecog hand gestures<sup>[[1]](#ref-1)</sup> and FER2013 emotion<sup>[[2]](#ref-2)</sup>.

The design is detect, crop, classify. Every 100 ms the app grabs the screen with `mss`, finds faces with an OpenCV SSD and hands with MediaPipe, strips the background off each hand, and passes small grayscale crops to three small Keras CNNs. Detection removes most of the image, which is why the classifiers can be small. **The gesture model gets 2,795 of 2,800 test images right (99.8%), the ASL model about 98%, and the emotion model 68.2% on FER2013.**

## Why It Matters

The goal was to help deaf and hard-of-hearing people and hearing people talk on ordinary video calls. A tool that needs its own camera feed doesn't fit that. We wanted a small layer that floats above whatever app you already use and labels what it sees.

That led to two design choices that shape everything else. Capture is a screen grab, so the tool needs no camera access and works on any conferencing app. And the app runs several classifiers on the same frame every 100 ms, so each one has to be small.

The scope was three recognizers. Two of them share a hand crop, with ASL finger-spelling classifying it into a letter and a hand-gesture model classifying the same crop into one of 10 gestures, while the third, an emotion model, classifies a face crop into one of the seven FER2013 labels.

The three of us built it as one team and share equal credit on the paper. I wrote the research paper and made its confusion-matrix and loss-curve figures, along with an early FER2013 notebook whose data pipeline and training recipe the emotion script kept and, later, the dataset downloader. Jalen wrote the model training scripts and the README, and Jayson wrote the desktop app.

## Technical Details

### Small CNNs on Cropped Inputs

None of the models is exotic. Each is a convolutional classifier that takes a small grayscale crop and outputs a softmax over classes, trained with Adam<sup>[[3]](#ref-3)</sup>. The emotion network is the deepest, with four convolution blocks with batch normalization<sup>[[4]](#ref-4)</sup> and dropout, ending in global average pooling. The ASL network is three convolutions and a dense layer with dropout. The gesture network is the plainest, one convolution and one dense layer with no dropout or batch normalization.

### Detect, Crop, Classify

The app never classifies a whole screenshot. A detector finds a face or a hand, the app crops around it, and a classifier sees only the crop. Faces come from OpenCV's DNN module running a pretrained Caffe single-shot detector<sup>[[5]](#ref-5)</sup>, keeping detections with confidence above 0.5. Hands come from MediaPipe Hands<sup>[[6]](#ref-6)</sup>, which returns 21 landmarks per hand. The app takes the min and max of the landmark coordinates as a bounding box and adds a 20 pixel margin. This is why the classifiers can be small. Detection has already removed most of the image.

### The Overlay

The app is a set of threads around a Qt window. Two worker threads each grab a region of the screen with `mss`, run their models, and emit a list of boxes and labels to the overlay widget, which paints them.

<figure class="excal" data-diagram="signifyai-architecture"><a href="/img/diagrams/signifyai-architecture.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/signifyai-architecture.webp" alt="SignifyAI screen overlay: two threads grab a screen region with mss every 100 ms, the face thread runs an OpenCV DNN face detector and a 48 by 48 emotion CNN, the hand thread runs MediaPipe Hands, a 20 pixel crop, rembg and a 128 by 128 grayscale image into a 10-class gesture CNN and a 29-class ASL CNN, and a frameless, always-on-top, mouse-transparent Qt overlay shows the results." width="2400" height="2482" loading="lazy" decoding="async"></a></figure>

The overlay is a frameless, always-on-top, mouse-transparent window, so clicks pass through it to the application underneath. A separate draggable toolbar toggles the face and hand threads and snaps to screen edges. There are three app variants. `main.py` does face and emotion only, `gesture.py` is the primary app with hand gestures, ASL and emotion, and `hands_no_gesture.py` is a debug variant.

### Where the Parameters Live

The three networks differ a lot in shape, and the parameter counts are not what the layer counts suggest.

<figure class="excal" data-diagram="signifyai-models"><a href="/img/diagrams/signifyai-models.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/signifyai-models.webp" alt="Layer stacks of the three SignifyAI CNNs: the gesture CNN has one 16-filter conv and a Dense 32 holding 2,097,184 of its 2,097,674 parameters, the ASL CNN has three conv stages and a Dense 128 holding 3,211,392 of its 3,307,805, and the emotion CNN has four batch-normalized conv blocks and global average pooling for 914,151 parameters." width="2400" height="2952" loading="lazy" decoding="async"></a></figure>

The gesture network has a single 16-filter convolution and yet holds 2,097,674 parameters, because the `Flatten` layer hands a 64 by 64 by 16 feature map, 65,536 values, to a `Dense(32)`, and that one layer holds 2,097,184 of them. The emotion network is by far the deepest and has the fewest parameters, 914,151, because global average pooling replaces the flatten.

## Implementation

### Screen Capture and the Hand Thread

The hand thread is Jayson's code. It grabs the screen, downscales it so hands look larger to the detector, runs MediaPipe, and then processes each hand in turn.

```python
with mss.mss() as sct:
    while self.running:
        sct_img = sct.grab(self.monitor)
        frame = np.array(sct_img)
        ...
        results = self.hands.process(rgb_frame)
        ...
        self.handsDetected.emit(hand_boxes)
        self.msleep(100)
```

The 100 ms sleep caps the loop at roughly ten iterations a second before the cost of the models themselves.

### One Crop, Two Classifiers

Each hand crop is resized to 256 by 256 and goes through `rembg`<sup>[[7]](#ref-7)</sup>, is composited over black using the alpha mask, resized back to the original crop size, converted to grayscale, resized to 128 by 128 and scaled to [0, 1]. Then the same array goes to both the gesture CNN and the ASL CNN.

```python
gesture_preds = self.gesture_model.predict(input_img)
...
asl_preds = self.asl_model.predict(input_img)
...
combined_label = f"Gesture: {gesture_name} | ASL: {asl_name}"
```

Every hand gets both a gesture and a letter, and the overlay shows the two side by side.

### The Face Thread

The face thread runs the Caffe SSD on a frame scaled by 0.5, keeps detections above 0.5 confidence, crops the face, converts it to 48 by 48 grayscale and runs the emotion CNN. It is the simpler of the two threads.

### Training the Models

Jalen's `facial_expression.py` trains the emotion network. It keeps the split logic, augmentation settings, Adam rate and early stopping from my early notebook, and replaces my architecture (Flatten and three Dense blocks) with Jalen's convolutional one. It uses the predefined FER2013 splits: 28,709 training images, 3,589 in `PublicTest` for validation and 3,589 in `PrivateTest` for the test. Training uses augmentation (rotation 10 degrees, shifts and zoom of 0.1, horizontal flip), Adam at 0.001, batch 64 for up to 100 epochs, early stopping with patience 10 on validation loss with the best weights restored, and a learning-rate reduction of 0.5 after 5 stalled epochs.

The gesture script holds out 30% of the images for validation and 20% of the rest for test, so 56% train, 30% validation and 14% test, at batch 32. The ASL script splits the alphabet set 80/20 with `image_dataset_from_directory` and seed 123 and trains for 30 epochs at batch 32. Our original ASL run used 36 classes, digits 0 to 9 and letters A to Z. The current ASL code covers 29, A to Z plus `del`, `nothing` and `space`.

### Datasets on One Command

My later contribution was `data/dataset.py`, a 58-line script that uses `kagglehub` to download the three datasets and symlinks them into `data/`, so every training script reads from a fixed path.

```python
DATASETS = {
    "asl_dataset":   {"handle": "grassknoted/asl-alphabet", ...},
    "fer2013.csv":   {"handle": "deadskull7/fer2013", ...},
    "leapGestRecog": {"handle": "gti-upm/leapgestrecog", ...},
}
```

## Problems

### 1. Hands on a Screen Are Small

A webcam demo gets a hand that fills a good part of the frame. A screen grab of a video call gets a hand inside a window inside a monitor. **The fix was to downscale the frame before detection so hands look larger to MediaPipe, and to run the detector permissively.** MediaPipe is configured for up to 10 hands with detection and tracking confidence of 0.3, a low threshold that trades false positives for recall.

### 2. Training Images Don't Look Like Screen Crops

LeapGestRecog frames are clean images of a hand on a plain background. A hand cropped from a real screen sits on a messy one. **That is why the hand thread removes the background before classifying.** It runs `rembg` on each crop, composites what is left over black, and converts it to grayscale, so the classifier sees the hand and not the room behind it.

### 3. Several Models Inside One 100 ms Loop

Each hand goes through `rembg` and two `predict` calls, and each face through the SSD and the emotion CNN, all on a loop that wakes every 100 ms. **Three choices keep the per-frame work small: faces and hands run on separate threads, detection shrinks every input to a crop, and the classifiers stay small.** Global average pooling is what keeps the deepest of them, the emotion network, under a million parameters.

### 4. An Overlay That Doesn't Get in the Way

A layer that sits over a video call is useless if it blocks the call. **The window is frameless, always on top and mouse-transparent, so every click lands on the app underneath.** The controls live in a separate draggable toolbar that snaps to screen edges and turns the face and hand threads on and off.

## Experiments

For this write-up I did two things.

1. I rebuilt the three layer stacks in TensorFlow 2.21 on Python 3.12 and counted parameters with `count_params()`, using `results/count_params.py`. The counts are in `results/param_counts.txt`.
2. I recounted the ASL, gesture and emotion accuracies from the confusion-matrix images. I read the cell values by eye from `confusion_1.png`, `confusion_2.png` and `confusion_3.png`, summed the diagonals, and checked the row sums against the known test-set sizes. The gesture matrix has 280 samples in each of 10 classes, and the emotion matrix rows sum to 3,589, the size of the FER2013 `PrivateTest` split. The arithmetic is in `results/confusion_arithmetic.txt`.

## Results

### Parameter Counts

<figure data-figure="chart:projects/signifyai/signifyai-params"></figure>

The three networks have 2,097,674 (gesture), 3,307,805 (ASL, 29 classes) and 914,151 (emotion, of which 912,103 are trainable) parameters. The 36-class ASL version has 3,308,708, a difference that sits entirely in the output layer. **The network with one convolution has more than twice the parameters of the one with four convolution blocks.**

### Gesture Accuracy

The gesture confusion matrix has 280 test samples in each of 10 classes, 2,800 in total. The diagonal is 279, 279, 280, 280, 280, 277, 280, 280, 280 and 280, which sums to 2,795. **The model got 5 wrong, an accuracy of 99.82%.** The training curve in the same figure has validation loss near 0.14 at the first epoch and close to zero by epoch 4. **LeapGestRecog is an easy dataset to fit**, with clean hands on plain backgrounds, and the model learns it in a few epochs.

### Emotion Accuracy

The emotion matrix rows sum to 3,589. The diagonal is 307, 25, 215, 775, 337, 332 and 457, which sums to 2,448. **That is 68.21%, typical of a small CNN on FER2013.** On the training curve, training loss falls from about 2.1 to about 0.86 and validation loss from about 1.98 to about 0.92, stopping around epoch 62.

<figure data-figure="chart:projects/signifyai/signifyai-emotion-recall"></figure>

Happy is the best class, at 775 of 879. Fear is the worst of the large classes, at 215 of 528, and is confused mostly with Sad (104), Angry (72), Neutral (61) and Surprise (58). Disgust has only 55 test samples, of which 25 are right, so its recall is a noisy estimate. Sad and Neutral are also confused, with 126 of 594 Sad images predicted Neutral. **An overall accuracy of 68% hides a model that gets Fear right only about 41% of the time** and mistakes it for Sad, Angry, Neutral or Surprise most of the rest.

### ASL Accuracy

The ASL confusion matrix is from our original 36-class run, with roughly 7 to 20 samples per class. I hand-counted it the same way. The diagonal sums to 493 and the off-diagonal cells to about 10, out of about 503 samples, so **roughly 98.0%.** The errors are single counts apart from 0 predicted as O (2 times) and V predicted as 2 (2 times), plus single errors such as A and S, and T and U. **Those are the pairs that look alike in a still image.**

## What I Would Change

### Study the Face Properly

FER2013 labels are a coarse proxy for what a signer's face is doing. In signed languages the face carries grammar as well as emotion. I would design an experiment around that directly instead of reusing seven emotion labels.

## References

1. <span id="ref-1"></span>Marin, G., Dominio, F., Zanuttigh, P. *Hand Gesture Recognition with Leap Motion and Kinect Devices*. IEEE International Conference on Image Processing (ICIP), 2014. LeapGestRecog is distributed on Kaggle as [gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).
2. <span id="ref-2"></span>Goodfellow, I. J., et al. *Challenges in Representation Learning: A Report on Three Machine Learning Contests*. Neural Networks 64, 2015. [arXiv:1307.0414](https://arxiv.org/abs/1307.0414)
3. <span id="ref-3"></span>Kingma, D. P., Ba, J. *Adam: A Method for Stochastic Optimization*. ICLR, 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
4. <span id="ref-4"></span>Ioffe, S., Szegedy, C. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML, 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
5. <span id="ref-5"></span>Liu, W., et al. *SSD: Single Shot MultiBox Detector*. ECCV, 2016. [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)
6. <span id="ref-6"></span>Zhang, F., et al. *MediaPipe Hands: On-device Real-time Hand Tracking*. 2020. [arXiv:2006.10214](https://arxiv.org/abs/2006.10214)
7. <span id="ref-7"></span>Gatis, D. *rembg: Rembg is a tool to remove images background*. Software, [github.com/danielgatis/rembg](https://github.com/danielgatis/rembg)
