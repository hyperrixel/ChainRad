# Inspiration

Healthcare system is usually on the bottom section of any government’s budget. Hospitals struggle from underfunding. Doctors, especially radiologists are overwhelmed. The COVID-19 pandemic gives a lot of extra pressure on the healthcare system and on hospitals’ staff as well. We want to create a solution that helps to handle the available resources especially human resources and work time better. We know, everybody can train a deep learning model. That’s why we bring this hackathon an idea that is not a “yet another solution” 

We are really devoted to cutting edge technologies and we are interested in healthcare.

We coded a **working demo** with a UI to show our vision. We made **EDA** to show we understand the data and what is usually wrong with the publicly available datasets.We wrote an **idea validation plan** to show what we can make with low end GPUs from poor data within a limited time. This validation plan has a lot of similarities with the necessary contents of *FDA 510(k) submission*. We planned **short and long time ranges** to bring out the most from the idea nut just in the long term, but in a short time period as well.

# What it does

ChainRad is a deep learning based radiology software to assist radiologists by returning the possibility of trained diseases from multiple neural network chains simultaneously. This sounds simple and not a real unique solution, but the real benefits are under the hood.

The neural network is built on very common pretrained models. We implemented 4 of them: 
- VGG-16 batchnorm
- ResNet 152
- DenseNet 161
- GoogleNet

We used those models as a headless model which means we cutted the last classification layers from them and used only the conculutional parts. 

These models are trained on regular images which are different then an image from the radiology department. That’s why we want to train our own pretrained network in long term. So it’s basically a residual network for healthcare images. However, this is the long term plan.  We used common models to validate our idea. 

We use a neural-network chain where each neural network is trained to identify only one disease.In our example we trained 14 different neural networks based on the same structure. Any incoming image moves through the preprocessing phase and the headless pretrained models before it reaches our classification layers. Since the headless models are pretrained, and we didn’t retrained them, each of our models is different from the other by the last classification layers. We trained only them. We think, it is better to use one-disease models instead of a giant monster-network, since each of them can learn better with less noise. 

A final software assists radiologists by giving the possibilities of trained diseases. However, this software does not contact the patients or with the patient's internal organs, nervous- or cardiovascular system, it’s a diagnostic tool for internal organs.


# Accomplishments that we're proud of
-  working demo with UI
- EDA
- unit test for data preparation
- 14 trained models
- validation plan

#What's next for ChainRad

We are thinking this state of ChainRad is rather a proof of concept than a production ready software though it has significant performance at the moment as well. In case if ChainRad would have a chance to get developed further, we would re-train the famous computer vision architectures from zero to get more healthcare-compatible  networks.
