### Rationale for Stratified 70/15/15 Split

The 70/15/15 split (train/validation/test) is a standard practice in machine learning to ensure robust model development while addressing dataset imbalances, as observed in the Reddit data (42.5% positive, 35.3% neutral, 22.2% negative sentiments).

#### Why Stratified Splitting?
- **Preserves Class Balance**: Without stratification, random splits risk uneven distributions across sets (e.g., test set might underrepresent negatives, skewing evaluation). Stratification enforces proportional representation in each set, maintaining the original ~42/35/22 ratio. This is implemented via `stratify=df['category']` in scikit-learn's `train_test_split`, enhancing reliability for imbalanced classification.

#### Why 70/15/15 Ratios?
- **Train (70%)**: Provides sufficient volume (~26,000 samples here) for model training, minimizing underfitting while allowing experimentation (e.g., with TF-IDF or embeddings).
- **Validation (15%)**: Allocates ~5,600 samples for iterative tuning without depleting training data.
- **Test (15%)**: Reserves ~5,600 untouched samples for final, unbiased performance assessment (e.g., F1-score), simulating real-world YouTube inference.
- **Rationale**: This ratio balances data efficiency for smaller datasets like ours; larger datasets might use 80/10/10. It supports scalability in MLOps (e.g., quick hyperparameter sweeps) and adaptability for future YouTube data augmentation.

#### Why Create a Validation Set?
- **Hyperparameter Tuning and Model Selection**: During modeling (e.g., in `train_model.py`), the validation set evaluates candidates (e.g., Logistic Regression vs. LSTM) and tunes parameters (e.g., via grid search) without "peeking" at test data, preventing overfitting and data leakage.
- **Early Feedback Loop**: Enables rapid iteration in CRISP-DM's Modeling phase—train on train set, validate on val set, refine—while keeping test set pristine for deployment decisions (e.g., AWS threshold).
- **Practical Benefit**: In production, it mirrors A/B testing; without it, you'd risk optimistic biases, undermining reliability.

This setup aligns with MLOps best practices, fostering maintainable pipelines. Consider k-fold cross-validation on the train set if validation proves noisy.