# Penguin Species Clustering Project

## Objective
Support a research team in Antarctica to identify different species of penguins using unsupervised learning. The dataset includes physical characteristics of penguins but lacks species labels. The goal is to use clustering to uncover natural groupings (species) in the data.

## Data Source
Data were collected by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.

## Dataset Description
The dataset `penguins.csv` contains the following columns:

| Column             | Description                          |
|--------------------|--------------------------------------|
| culmen_length_mm   | Culmen length in millimeters         |
| culmen_depth_mm    | Culmen depth in millimeters          |
| flipper_length_mm  | Flipper length in millimeters        |
| body_mass_g        | Body mass in grams                   |
| sex                | Sex of the penguin (Male/Female)     |

## Approach
1. **Data Loading and Exploration**  
   Loaded and examined dataset structure.

2. **Preprocessing**  
   - Encoded categorical variables (e.g., `sex`) using one-hot encoding.  
   - Standardized all features using `StandardScaler`.

3. **K-Means Clustering**  
   - Determined the optimal number of clusters using the Elbow Method.  
   - Ran KMeans clustering with `n_clusters = 4`.

4. **Visualization & Summary**  
   - Visualized clusters using scatter plots.  
   - Summarized key statistics of each cluster.

## Outcome
- Created meaningful clusters of penguins based on body measurements and encoded sex.
- Allowed researchers to potentially differentiate among Adelie, Chinstrap, and Gentoo penguin species.

## Files in the Repository
- `penguin_clustering.py`: Python script with the complete analysis pipeline.
- `penguins.csv`: Input dataset.
- `README.md`: Project documentation.

## Libraries Used
- pandas  
- matplotlib  
- scikit-learn  

---

