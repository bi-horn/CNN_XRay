# Machine Learning - Convolutional Neural Networks with PyTorch

### Application of an AI System for Pneumonia Detection Using Chest X-Ray Images

Inspiration: Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C., Liang, H., Baxter, S. L. & Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. cell, 172(5), 1122-1131.

Dataset downloaded from: https://data.mendeley.com/datasets/rscbjbr9sj/2
5,855 chest X-ray images of patients with and without pneumonia; 2 classes Normal (1,582) and Pneumonia (4,273)

Because CUDA is not available for download on a MacOS system anymore, I used Google Colab since it already has the CUDA toolkit and drivers installed on its virtual machines

Goal 1: Comparison of a network with a deep neural structure (ResNet34) vs a simple self-defined neural network (2 convolutional layers and 3 fully connected layers) (CNN_XRay.ipynb)

Goal 2: Comparison of a pretrained and not pretrained network with a deep neural structure (ResNet34) (CNN_XRay_TransferLearning.ipynb)





