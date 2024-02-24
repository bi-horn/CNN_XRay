# Machine Learning - Convolutional Neural Networks with PyTorch

### Application of an AI System for Pneumonia Detection Using Chest X-Ray Images

Inspiration: Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C., Liang, H., Baxter, S. L. & Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. cell, 172(5), 1122-1131.

Dataset downloaded from: https://data.mendeley.com/datasets/rscbjbr9sj/2
5,855 chest X-ray images of patients with and without pneumonia; 2 classes Normal (1,582) and Pneumonia (4,273)

Because CUDA is not available for download on a MacOS system anymore, I used Google Colab since it already has the CUDA toolkit and drivers installed on its virtual machines
If working with Jupyter Notebook is preferred, an alternative version with slight customizations is available (CNN_XRay_Jupyter.ipynb).

#### Goal 1: Comparison of a network with a deep neural structure (ResNet34) vs a simple self-defined neural network (2 convolutional layers and 3 fully connected layers) (CNN_XRay.ipynb or CNN_XRay_Jupyter.ipynb)

The simple CNN model outperforms the more complex ResNet34 model in detecting Pneumonia in X-ray images. Despite its simplicity, the simple CNN achieves an accuracy of over 90% after just the third epoch, whereas the ResNet34 model achieves similar accuracy only after the fifth epoch.

Both models encounter difficulties in reducing the validation loss below ~15%. While the models are effective at accurately classifying images, further optimization may be necessary to improve their generalization ability and reduce overfitting.

Performance Differences Between Models: The results suggest that the more complicated model, while having fewer false positive cases, may not perform as well in correctly identifying positive cases. Conversely, the simpler model has a more balanced performance in terms of false positive and false negative cases.

Trade-off Between Complexity and Performance: It seems that the simpler model may exhibit more robust performance as it is less susceptible to false negative cases. This could indicate that a certain level of model complexity does not always lead to improved performance.


#### Goal 2: Comparison of a pretrained and not pretrained network with a deep neural structure (ResNet34) (CNN_XRay_TransferLearning.ipynb)

-The accuracy of the pre-trained ResNet34 model already exceeds 90 % after the third epoch ( similar to the simpler model ) 

-Nevertheless, the pre-trained ResNet34 model also encounters difficulties in minimizing the loss of the validation dataset, indicating possible limitations despite its high accuracy

-However, the pre-trained model shows significantly fewer false negative results on the test dataset (12 instead of 22), which is an improvement over the untrained ResNet34 model





