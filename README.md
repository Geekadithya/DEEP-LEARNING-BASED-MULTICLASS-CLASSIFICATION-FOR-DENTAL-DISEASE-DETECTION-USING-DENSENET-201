# DEEP-LEARNING-BASED-MULTICLASS-CLASSIFICATION-FOR-DENTAL-DISEASE-DETECTION-USING-DENSENET-201
This project uses DenseNet-201 for automated detection of five dental diseases, achieving 96% accuracy on a dataset of some intraoral images. This approach highlights deep learning’s potential in precise, scalable dental diagnostics, with plans for real-time clinical integration.
Dental diseases, including gingivitis, tooth discoloration, mouth ulcers, cavities, and calculus, are widespread and pose substantial risks to oral health. Early detection is crucial to prevent complications, yet traditional diagnostics largely depend on visual examination, which can be subjective and error-prone. This project leverages deep learning to address these challenges by developing an automated, multiclass classification model for dental disease detection using DenseNet-201 architecture.

DenseNet-201 was selected for its efficient feature propagation through dense connections, making it highly suitable for medical imaging tasks. The model was trained on a dataset of 1,500 intraoral images, with 300 images per class, covering five types of dental diseases. To ensure data consistency, preprocessing techniques such as resizing and normalization were applied, and a softmax activation function was used in the final layer to support multiclass classification.

The model achieved a classification accuracy of 96%, surpassing alternative models like InceptionResNetV2, which reached 94% under similar conditions. Performance evaluation was conducted using metrics such as accuracy, ROC curves, and AUC scores, confirming the model’s reliability in automated dental disease detection.

This study demonstrates the potential of deep learning, especially DenseNet-201, to enhance dental diagnostics by providing a scalable, precise solution for early disease identification. The model's high accuracy presents a viable alternative to manual diagnostics, contributing to timely and accurate diagnoses in clinical settings. Future work will focus on refining the model’s generalizability and robustness by testing it on more diverse datasets and integrating it into real-time diagnostic systems, ultimately aiming to assist dental professionals in improving patient outcomes across various healthcare environments.
