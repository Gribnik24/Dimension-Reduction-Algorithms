# Unsupervised Learning and Dimensionality Reduction

## Project Overview

This project explores **dimensionality reduction and low-rank matrix approximation** through three practical tasks: predicting user age from sparse book ratings, visualizing handwritten digits, and applying Singular Value Decomposition (SVD) to image compression and video background estimation.

The goal is to compare linear and nonlinear representations, evaluate their effect on model quality and computation time, and understand how a small number of singular components can preserve the main structure of images and video. The experiments use the **Book Recommendation Dataset**, **MNIST**, and a video from the **Background Models Challenge**.

## Dataset

### Book Recommendation Dataset

Source: [Book Recommendation Dataset on Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

- `Books.csv`: book metadata, including `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, and `Publisher`.
- `Ratings.csv`: user-book interactions with `User-ID`, `ISBN`, and `Book-Rating`.
- `Users.csv`: user information with `User-ID`, `Location`, and `Age`.

After preprocessing and merging, the user-book matrix contains **40,543 users**, **119,948 books**, and **269,621 nonzero ratings**. Only **0.0055%** of its entries are nonzero, so it is stored as a SciPy CSR matrix. The target variable is user age; despite the notebook's filename, this is a **regression** task.

### MNIST

Source: [MNIST in IDX format on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

The dataset contains 60,000 training and 10,000 test grayscale images of handwritten digits. Each image has a resolution of **28 × 28 pixels** and is represented by **784 features**. A stratified sample of **3,000 training images** is used to compare two-dimensional embeddings. Three randomly selected training images are used for SVD compression.

### Background Models Challenge

Source: [Background Models Challenge](https://backgroundmodelschallenge.eu/), `bmc_real.zip`.

The background estimation experiment uses `Video_008.avi`: **792 decoded frames**, **320 × 240 pixels**, and **10 FPS**. Grayscale frames form an uncentered matrix of shape **(76,800, 792)**, with one frame per column.

## Project Structure

```text
├── README.md                                  # Original project assignment
├── requirements.txt                           # Python dependencies
├── LICENSE                                    # School 21 license
└── src/
    ├── README.md                              # Project overview and results
    ├── 1_questions.ipynb                       # Curse of dimensionality, PCA, SVD, NMF, LLE
    ├── 2_Classification_with_Sparse_Features.ipynb # Age regression on sparse book ratings
    ├── 3_SVD_Image_Compression.ipynb           # MNIST embeddings and image compression
    ├── 4_SVD_Background_Detection.ipynb        # Video background estimation
    ├── Books.csv                              # Local dataset files: download separately
    ├── Ratings.csv
    ├── Users.csv
    ├── Video_008.avi
    ├── train-images-idx3-ubyte/
    │   └── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte/
    │   └── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte/
    │   └── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte/
        └── t10k-labels-idx1-ubyte
```

## Methodology

1. **Age Prediction from Sparse Features** — [Classification_with_Sparse_Features.ipynb](Classification_with_Sparse_Features.ipynb)
   - Explored books, ratings, and users; corrected malformed book metadata and standardized text fields.
   - Removed zero ratings and users with missing age; capped ages above 100.
   - Merged the tables and built a CSR matrix using rating values as features, averaging duplicate user-book ratings.
   - Split users into train, validation, and test sets in a **60/20/20** ratio with.
   - Trained `LinearRegression`, `Lasso`, and `RandomForestRegressor`; tuned Lasso and Random Forest with `RandomizedSearchCV` and two-fold cross-validation.
   - Compared the original features with **50 PCA components**, **50 UMAP components**, and **3 t-SNE components** using validation RMSE, MAE, and fitting time.
   - Used the full training and validation sets for PCA, and subsets of **1,000 training / 500 validation users** for UMAP and t-SNE. The t-SNE embedding was fitted jointly on both subsets because it has no separate `transform` method.
   - Added a comparison of Lasso coefficients and the number of zeroed features before and after dimensionality reduction.

2. **MNIST Visualization** — [SVD_Image_Compression.ipynb](SVD_Image_Compression.ipynb)
   - Loaded raw IDX files, flattened the images, and normalized pixel values to `[0, 1]`.
   - Applied **PCA, SVD, Randomized-SVD, t-SNE, UMAP, and LLE** to the same stratified sample of 3,000 images.
   - Compared two-dimensional embeddings using **Silhouette score**, **Davies–Bouldin index**, **10-neighbor purity**, and fitting time.
   - Used digit labels only for visualization and evaluation. PCA centers the data; the SVD variants operate on the uncentered matrix.

3. **Image Compression with SVD** — [SVD_Image_Compression.ipynb](SVD_Image_Compression.ipynb)
   - Selected three MNIST images with a fixed random seed of 42 and decomposed each image matrix using `numpy.linalg.svd`.
   - Reconstructed images at ranks **1, 2, 3, 4, 5, 7, 10, 14, 20, and 28**.
   - Plotted reconstructions, singular value spectra, and cumulative retained energy; measured reconstruction RMSE.
   - Determined the number of modes needed to preserve **95%** and **99%** of the squared pixel energy.

4. **Video Background Estimation with SVD** — [SVD_Background_Detection.ipynb](SVD_Background_Detection.ipynb)
   - Converted all decoded frames to grayscale and stacked them into an uncentered pixel-by-frame matrix.
   - Computed the reduced SVD and reconstructed the first frame at ranks **1, 2, 5, 10, and 20**.
   - Compared the original frame, the rank-1 background estimate, and the absolute residual.

## Key Findings

The values below summarize **saved notebook outputs**. Fitting times and results can vary with hardware, library versions, and random initialization.

### 1. Sparse Age Regression

Validation results for models trained on the full training set:

| Model | Original features: RMSE | Original features: MAE | PCA: RMSE | PCA: MAE |
|---|---:|---:|---:|---:|
| Linear Regression | 28.1314 | 17.6911 | 14.4619 | 11.5122 |
| Lasso | 14.5113 | 11.5583 | 14.5003 | 11.5477 |
| Random Forest | 14.5157 | 11.5621 | **14.0959** | **11.1263** |

- **Random Forest with PCA** achieved the lowest validation RMSE and MAE among the configurations evaluated on the full validation set.
- Reducing the representation from 119,948 book features to 50 PCA components substantially improved the unregularized linear model's validation error.
- The recorded training and search time for the three regressors fell from **161.20 s** on the original features to **34.90 s** after PCA, with **1.59 s** spent on dimensionality reduction.
- UMAP and t-SNE regression experiments used smaller samples, so their scores and training times are not directly comparable with the full-data results. Joint t-SNE fitting also uses validation feature information and does not measure performance on independently embedded unseen users.
- The notebook creates a held-out test set, but the reported comparisons use the **validation set**.

### 2. MNIST Embedding Comparison

| Method | Silhouette ↑ | Davies–Bouldin ↓ | 10-neighbor purity ↑ | Fitting time (s) |
|---|---:|---:|---:|---:|
| t-SNE | 0.2477 | 2.7006 | **0.8599** | 17.8234 |
| UMAP | **0.3186** | **1.6124** | 0.8562 | 67.5880 |
| PCA | 0.0137 | 5.3068 | 0.3658 | 3.2877 |
| LLE | -0.1367 | 10.3367 | 0.2878 | 5.1544 |
| SVD | -0.0411 | 6.0973 | 0.2798 | 1.8284 |
| Randomized-SVD | -0.0411 | 6.0973 | 0.2798 | **0.0481** |

- **t-SNE** produced the highest local class purity, while **UMAP** achieved the best Silhouette score and lowest Davies–Bouldin index for this sample and these settings.
- **Randomized-SVD** reproduced the reported class-separation metrics of full SVD while taking approximately **38 times less fitting time** in the saved run.
- Neighbor purity describes local separation within the embedded sample; it is not classification accuracy on unseen images.

### 3. SVD Image Compression

| MNIST image index | Digit | Rank for 95% energy | Rank for 99% energy |
|---|---:|---:|---:|
| 39,274 | 5 | 4 | 8 |
| 5,354 | 3 | 5 | 8 |
| 46,436 | 0 | 5 | 8 |

- All three selected images retained **99% of their squared pixel energy with 8 singular modes**.
- For a 28 × 28 image, storing the rank-8 factors requires **456 scalar values**, compared with **784** original pixel values. This compares element counts, not encoded file sizes or byte usage.
- Because the images are not centered, the singular-value energy ratio measures **retained energy**, rather than variance around the mean as in PCA.

### 4. Video Background Estimation

- The leading singular component retained **98.09%** of the video matrix energy.
- **Rank 1** provided a visually useful background estimate, suppressing the two vehicles in the first frame while preserving the persistent scene.
- Higher ranks reduced first-frame reconstruction RMSE from **0.0530** at rank 1 to **0.0396** at rank 20, but also restored more scene variation and possible foreground traces. Lower reconstruction error does not necessarily indicate a better background estimate.
- The absolute residual highlights differences from the background approximation; it is not a binary foreground mask. Background quality was assessed visually without a ground-truth background.

## Conclusion

The experiments demonstrate how **dimensionality reduction serves different goals**: PCA produces compact features for regression, nonlinear embeddings reveal digit-class structure in two dimensions, and low-rank SVD approximations retain the dominant content of images and video.

In the saved runs, PCA improved age-regression results while reducing training time, t-SNE and UMAP gave the strongest MNIST class separation, and a small number of singular modes preserved most image and video energy. These findings apply to the evaluated datasets, samples, and settings; each task requires its own quality measure.

## License

Datasets are subject to their respective providers' terms of use.
