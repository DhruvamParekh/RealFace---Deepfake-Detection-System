# RealFace---Deepfake-Detection-System

Overview
RealFace is a machine learning-based system designed to detect deepfake videos by analyzing facial features and identifying telltale signs of tampering. The project focuses on practical, efficient, and interpretable deepfake detection methods by leveraging both handcrafted features and deep learning models.

As deepfake technology grows increasingly realistic and accessible, reliable detection tools like RealFace are essential for maintaining trust in visual media.

Problem Statement
Deepfake content—synthetic media created using AI to impersonate faces and voices—poses serious threats ranging from misinformation and fraud to privacy violations and reputational harm. With traditional methods falling short in identifying well-crafted deepfakes, there is a critical need for robust detection systems that can operate effectively on limited hardware and real-world data.

Methodology
1. Dataset Used
Due to computational constraints, the project uses the Small-scale Deepfake Forgery Video Dataset (SDFVD):

106 videos (53 real and 53 fake)

Fake videos created using Remaker AI

Videos trimmed to 4–5 seconds, 720p resolution

Source: Mendeley Dataset

2. Frame Extraction
To avoid the computational overhead of processing entire videos, keyframe extraction (I-Frames) is used to efficiently capture high-quality, self-contained frames. Benefits include faster processing, reduced redundancy, and clearer artifact detection.

3. Face Detection
MTCNN (Multi-task Cascaded Convolutional Networks) is used to extract facial regions from frames. This isolates the region of interest and removes irrelevant background, improving both speed and accuracy.

4. Feature Engineering
Two major types of features are extracted:

Texture Features: Local Binary Patterns (LBP) to quantify unnatural skin textures.

Contour Features: Canny edge detection to identify irregular or inconsistent facial outlines.

Each frame is represented by an 11-dimensional feature vector.

5. Model Training
Several classifiers were trained and tested:

Logistic Regression
Random Forest
XGBoost

A video is classified as fake if any frame within it is flagged as fake.

6. Improved Model via Transfer Learning
Xception CNN architecture, fine-tuned on extracted face frames.
Ensemble approach combines Xception’s output with the Random Forest’s prediction by averaging probabilities.

Evaluation
Model performance was assessed using:

Frame-level and video-level accuracy
Confusion matrix
Loss curves

Final predictions were based on frame-wise classification consensus to ensure sensitivity to even a single forged segment.

Literature Survey
Research was inspired and supported by literature on deepfake detection:

"Deepfakes Detection Techniques Using Deep Learning: A Survey" – Almars, 2021
DOI: 10.4236/jcc.2021.95003

Surveyed both traditional and modern approaches including CNNs, RNNs, LSTMs, GAN artifact analysis, and temporal and audio-visual consistency models.

Challenges Encountered
Hardware limitations prevented use of large-scale datasets like DFDC.
Generalization remains a key challenge—models trained on one dataset may not perform well on another.
Detection lag: real-time implementation and deployment needs further exploration.

Future Directions
Extend to real-time detection for social media platforms.
Use multimodal data (e.g., voice, heart rate signals) for enhanced accuracy.
Explore self-supervised or few-shot learning to reduce the dependency on large labeled datasets.
Integrate explainability modules to make detection interpretable to end-users.

Key Takeaways
Efficient deepfake detection is possible even with smaller datasets using a combination of handcrafted and deep learning features.
Transfer learning with fine-tuned CNNs significantly boosts performance.
Keyframe-based processing and facial region isolation are critical for model efficiency.
