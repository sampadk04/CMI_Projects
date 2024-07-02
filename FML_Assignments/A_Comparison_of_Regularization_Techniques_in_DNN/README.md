# A Comparison of Regularization Techniques in DNN

## Reading Project Report

Artificial Neural Networks (ANN) have garnered significant attention from researchers and engineers because of their ability to solve many complex problems with reasonable efficacy. When provided with sufficient data during the training process, ANNs can achieve excellent performance. However, if the training data are insufficient, the predefined neural network model may suffer from overfitting and underfitting problems. To address these issues, several regularization techniques have been devised and widely applied to ANNs. Despite this, choosing the most suitable scheme for ANNs remains challenging due to the lack of comparative information on the performance of these schemes when pitted against each other.

This paper presents a comparative study of several popular regularization techniques by evaluating the training and validation errors in a Deep Neural Network (DNN) model using a real-world weather dataset. The paper also validates the conclusions based on training and validation errors using an independent test set to determine the best regularization paradigm.

### Experiment Overview
The goal of the experiment is to compare the performance of several powerful regularization methods, including autoencoders, data augmentation, batch normalization, and L1 regularization. The experiment follows these steps:
1. The dataset is tested on a DNN without applying any regularization methods to establish a baseline for errors.
2. The same DNN model is then trained with each regularization technique applied, and the errors are calculated.
3. Each regularization scheme's performance is analyzed by comparing the training and validation errors.

The experiments were conducted on the following hardware and software setup:
- CPU: Intel Core i7-4790K (no GPU processing)
- RAM: 8 GB
- OS: Windows 10
- Programming Language: Python
- Libraries: TensorFlow, Sci-Kit Learn
- Visualization: Matplotlib

### Dataset and Model Architecture
The weather dataset used for training and validating the DNN was collected from a Korean government website. Each sample in the dataset has 35 features, representing seven weather parameters (average temperature, maximum temperature, minimum temperature, average wind speed, average humidity, cloudiness, and daylight hours) over five consecutive days. The target label is the average temperature of the next day.

The basic DNN model architecture included:
- Input Layer: 35 neurons
- Hidden Layers: 2 layers with 50 neurons each
- Output Layer: 1 neuron

### Regularization Techniques Compared
1. **Autoencoders:**
- Implemented to reduce noise from the input data by compressing and then decompressing the data.
- Resulted in underfitting due to high training and validation errors.

2. **Batch Normalization:**
- Standardized layer inputs to stabilize and speed up the training process.
- Showed better performance but exhibited slight overfitting.

3. **L1 Regularization:**
- Reduced the sensitivity of weights corresponding to irrelevant features.
- Still suffered from overfitting, with low training errors but high validation errors.

4. **Data Augmentation:**
- Expanded the training dataset by applying transformations.
- Two techniques were used: summing features and averaging features over consecutive days.
- The averaging technique achieved the best performance with good generalization and minimal overfitting.

### Results and Conclusion
The study found that some models using regularization techniques demonstrated better performance than those without any regularization methods. The data augmentation technique, especially the one based on averaging features, showed the best overall performance. The results indicated that this method provided the best balance between training and validation errors, leading to effective generalization of the model.

### Credits
- Aslah Ahmed Faizi
- Sampad Kumar Kar