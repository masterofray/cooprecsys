# Collaborative Filtering

## 1. What is Collaborative Filtering?
Collaborative Filtering (CF) is a fundamental technique in recommender systems that automates the process of predicting a user's preference or rating for a specific item based on the observed preferences of a community of other users. Unlike content-based filtering, which relies on the intrinsic properties of items (e.g., genre, color, keywords), collaborative filtering relies solely on past user-item interactions. The foundational assumption is that if User A and User B have historically agreed on the quality of several items, they are likely to agree on a new, unseen item in the future.

## 2. Is it about User-to-User, Item-to-Item, or User-to-Item Similarity?
Collaborative Filtering encompasses **all three perspectives**, depending on the specific algorithmic approach adopted:
- **User-to-User Similarity (Memory-Based CF):** This approach directly compares the rating vectors of users. If the system needs to recommend an item to User $u$, it finds $k$ other users whose rating history is most similar to $u$'s (neighbors).
- **Item-to-Item Similarity (Memory-Based CF):** This approach transposes the matrix and compares items based on how users have rated them. If a user liked Item $i$, the system recommends other items that have the most similar rating profile to $i$. This was famously popularized by Amazon's "Customers Who Bought This Item Also Bought" feature.
- **User-to-Item Relationship (Model-Based CF / Matrix Factorization):** This approach does not compute explicit similarity between raw rows or columns. Instead, it models the interaction as an inner product of latent factors—a **User-to-Item** affinity. It discovers abstract features that explain the observed ratings, such that the predicted rating $\hat{r}_{ui}$ is a function of the user's preference vector and the item's characteristic vector.

## 3. When Exactly is the Proper Case and Time to Use Collaborative Filtering?
Collaborative Filtering is the appropriate methodology under the following specific conditions:
- **Sparse User Feedback:** You have a large user base and a large item catalog, but each individual user has only interacted with a tiny fraction of the total items.
- **Subjective or Aesthetic Domains:** The items lack objective, machine-readable attributes (e.g., movies, music, jokes, handmade crafts). For example, you cannot analyze the "pixel data" of a movie to determine if it is "funny"; you must rely on what other people who laughed at the same things thought.
- **Cross-Category Discovery:** You want to surprise the user with recommendations from categories they have never visited (serendipity). For instance, a user who likes *Blade Runner* might like a specific Philip K. Dick novel, an association only visible through overlapping fanbases, not through text analysis.

## 4. Where Can We Implement Collaborative Filtering?
Collaborative Filtering is implemented across a wide spectrum of digital services where user decision fatigue is high:
- **E-commerce Platforms:** Amazon, eBay, and Etsy use it for product recommendations on detail pages and checkout cross-sells.
- **Streaming Media Services:** Netflix (movie suggestions), Spotify (Discover Weekly playlist), and YouTube (video recommendations) rely heavily on CF to keep users engaged by surfacing deep catalog content.
- **Social Networks:** LinkedIn ("People You May Know"), Facebook (Friend Suggestions), and Twitter (Who to Follow) treat "following a user" as the item interaction.
- **Digital Advertising:** Ad exchanges use CF to predict Click-Through Rate (CTR) for a specific user-ad pair based on similar user cohorts.

## 5. Why Should We Use Collaborative Filtering?
The primary justification for using Collaborative Filtering is **domain independence and discovery**. We should use it because:
1.  **No Feature Engineering Required:** It eliminates the need for subject matter experts to manually tag every song with "distorted guitar" or every book with "unreliable narrator." The system learns these latent descriptors from the data itself.
2.  **Quality of Insight:** It captures nuanced, hard-to-quantify preferences (e.g., "Movies with a bittersweet ending").
3.  **Scalability of Model Training:** Modern matrix factorization algorithms are highly parallelizable and can handle matrices with hundreds of millions of rows and columns efficiently using Stochastic Gradient Descent (SGD) or Alternating Least Squares (ALS).

## 6. Step-by-Step Mathematical Workflow with Formulation
When we have the data, specifically a set of triplets $(u, i, r_{ui})$ where $u$ is user ID, $i$ is item ID, and $r_{ui}$ is the interaction strength (explicit rating or implicit count). We work with **Model-Based Collaborative Filtering (Matrix Factorization)** as follows:

### **Step 1: Define the Interaction Matrix $R$**
We construct a sparse matrix $R \in \mathbb{R}^{m \times n}$, where $m$ is the number of users and $n$ is the number of items. Most entries $r_{ui}$ are missing (NaN).

### **Step 2: Mathematical Model Definition (Hypothesis)**
We assume each user $u$ can be represented by a latent vector $p_u \in \mathbb{R}^k$ and each item $i$ by a latent vector $q_i \in \mathbb{R}^k$. Here, $k$ is the number of latent factors (dimensionality of the abstract space, e.g., $k=50$ or $100$).
The predicted rating $\hat{r}_{ui}$ is modeled as the dot product (interaction) between the user and item vectors:
$$\hat{r}_{ui} = q_i^T p_u = \sum_{f=1}^{k} q_{if} \cdot p_{uf}$$

### **Step 3: Define the Loss Function (Objective)**
We need to minimize the difference between the observed ratings $r_{ui}$ and the predicted ratings $\hat{r}_{ui}$. To prevent overfitting to the sparse data, we introduce **L2 Regularization**. The objective function $\mathcal{L}$ to minimize is:
$\mathcal{L} = \min_{p^*, q^*} \sum_{(u,i) \in \mathcal{K}} \left( r_{ui} - q_i^T p_u \right)^2 + \lambda \left( \|q_i\|^2 + \|p_u\|^2 \right)$

Where:
- $\mathcal{K}$ is the set of known (user, item) pairs in the training data.
- $\lambda$ is the regularization hyperparameter controlling the penalty on vector magnitude.

### **Step 4: Optimization via Stochastic Gradient Descent (SGD)**
We iterate over each known rating $r_{ui}$ in the training set. We compute the prediction error:
$e_{ui} = r_{ui} - q_i^T p_u$

We then compute the gradients of the loss with respect to the parameters and update them in the opposite direction of the gradient. **Update for User Vector $p_u$:**
$p_u \leftarrow p_u + \eta \cdot (e_{ui} \cdot q_i - \lambda \cdot p_u)$

**Update for Item Vector $q_i$:**
$q_i \leftarrow q_i + \eta \cdot (e_{ui} \cdot p_u - \lambda \cdot q_i)$
Where $\eta$ is the learning rate.

### **Step 5: Prediction for Missing Values**
After convergence (or a fixed number of epochs), we reconstruct the full matrix $\hat{R}$. For a user $u$ and an unseen item $j$, the final predicted score is:
$\hat{r}_{uj} = q_j^T p_u$
We then recommend the top-$N$ items with the highest $\hat{r}_{uj}$ values that the user has not yet interacted with.
