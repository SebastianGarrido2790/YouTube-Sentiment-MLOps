"""
Feature Engineering Component

Transforms cleaned text into numerical features using TF-IDF or DistilBERT,
combined with domain-specific derived metrics.
"""

import pickle
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.constants import FEATURES_DIR, TEST_PATH, TRAIN_PATH, VAL_PATH
from src.entity.config_entity import FeatureEngineeringConfig
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="feature_engineering_component")


class FeatureEngineering:
    """
    Handles the transformation of cleaned text into numerical feature matrices.

    Supports both traditional TF-IDF vectorization and transformer-based
    DistilBERT embeddings, combined with domain-specific derived features.
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """
        Initializes the FeatureEngineering component.

        Args:
            config (FeatureEngineeringConfig): Configuration defining feature strategies
                and hyperparameters.
        """
        self.config = config

    def get_distilbert_embeddings(
        self, texts: list[str], device: str | None = None, batch_size: int = 32
    ) -> np.ndarray:
        import torch
        from transformers import DistilBertModel, DistilBertTokenizer

        logger.info("Loading DistilBERT tokenizer and model...")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        embeds = []
        logger.info(f"Generating DistilBERT embeddings on device: {device} (batch size: {batch_size})")

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
                    device
                )

                outputs = model(**inputs)
                pooled_output = outputs.last_hidden_state.mean(dim=1)
                embeds.append(pooled_output.cpu().numpy())

        logger.info("✅ DistilBERT feature generation complete.")
        return np.vstack(embeds)

    @staticmethod
    def build_derived_features(df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["char_len"] = df["clean_comment"].str.len()
        df["word_len"] = df["clean_comment"].str.split().str.len()

        pos_words: set[str] = {"good", "great", "love", "like", "positive", "best"}
        neg_words: set[str] = {"bad", "hate", "worst", "negative", "shit", "fuck"}

        def count_lexicon_ratio(text: str, lexicon: set[str]) -> float:
            words = text.split()
            return len([w for w in words if w in lexicon]) / max(len(words), 1)

        df["pos_ratio"] = df["clean_comment"].apply(lambda x: count_lexicon_ratio(x, pos_words))
        df["neg_ratio"] = df["clean_comment"].apply(lambda x: count_lexicon_ratio(x, neg_words))

        return df[["char_len", "word_len", "pos_ratio", "neg_ratio"]].values

    def build_features(self) -> None:
        logger.info("Initializing Feature Engineering")
        logger.info(f"Targeting outputs to: {FEATURES_DIR}")

        # Ensure directory exists
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        test_df = pd.read_parquet(TEST_PATH)

        train_texts = train_df["clean_comment"].fillna("").tolist()
        val_texts = val_df["clean_comment"].fillna("").tolist()
        test_texts = test_df["clean_comment"].fillna("").tolist()

        logger.info("Building derived numerical features...")
        X_train_derived = FeatureEngineering.build_derived_features(train_df)
        X_val_derived = FeatureEngineering.build_derived_features(val_df)
        X_test_derived = FeatureEngineering.build_derived_features(test_df)

        if self.config.use_distilbert:
            logger.info("Using DistilBERT Strategy!")
            X_train_text = self.get_distilbert_embeddings(train_texts, batch_size=self.config.distilbert_batch_size)
            X_val_text = self.get_distilbert_embeddings(val_texts, batch_size=self.config.distilbert_batch_size)
            X_test_text = self.get_distilbert_embeddings(test_texts, batch_size=self.config.distilbert_batch_size)
        else:
            logger.info("Using TF-IDF Strategy!")
            try:
                min_n, max_n = map(int, self.config.best_ngram_range.strip("()").split(","))
            except Exception as e:
                logger.warning(f"Could not parse best_ngram_range: {e}. Falling back to (1,2)")
                min_n, max_n = 1, 2

            vectorizer = TfidfVectorizer(
                ngram_range=(min_n, max_n),
                max_features=self.config.best_max_features,
            )
            X_train_text = vectorizer.fit_transform(train_texts)
            X_val_text = vectorizer.transform(val_texts)
            X_test_text = vectorizer.transform(test_texts)

            vectorizer_path = FEATURES_DIR / "vectorizer.pkl"
            with open(vectorizer_path, "wb") as f:
                pickle.dump(vectorizer, f)
            logger.info(f"Saved TF-IDF vectorizer to {vectorizer_path}")

        X_train_final = (
            hstack([X_train_text, X_train_derived])  # type: ignore
            if issparse(X_train_text)
            else np.hstack([X_train_text, X_train_derived])  # type: ignore
        )
        X_val_final = (
            hstack([X_val_text, X_val_derived]) if issparse(X_val_text) else np.hstack([X_val_text, X_val_derived])  # type: ignore
        )
        X_test_final = (
            hstack([X_test_text, X_test_derived]) if issparse(X_test_text) else np.hstack([X_test_text, X_test_derived])  # type: ignore
        )

        if not issparse(X_train_final):
            X_train_final = csr_matrix(X_train_final)
            X_val_final = csr_matrix(X_val_final)
            X_test_final = csr_matrix(X_test_final)

        save_npz(FEATURES_DIR / "X_train.npz", X_train_final)
        save_npz(FEATURES_DIR / "X_val.npz", X_val_final)
        save_npz(FEATURES_DIR / "X_test.npz", X_test_final)
        logger.info("Saved X feature matrices (.npz)")

        le = LabelEncoder()

        train_encoded = le.fit_transform(train_df["category_encoded"])
        val_encoded = le.transform(val_df["category_encoded"])
        test_encoded = le.transform(test_df["category_encoded"])

        # Verify classes
        expected_classes = [0, 1, 2]
        if list(cast(Any, le.classes_)) != expected_classes:
            logger.warning(f"LabelEncoder classes {le.classes_} do not match expected {expected_classes}")

        np.save(FEATURES_DIR / "y_train.npy", cast(Any, train_encoded))
        np.save(FEATURES_DIR / "y_val.npy", cast(Any, val_encoded))
        np.save(FEATURES_DIR / "y_test.npy", cast(Any, test_encoded))
        logger.info("Saved y label arrays (.npy)")

        le_path = FEATURES_DIR / "label_encoder.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(le, f)
        logger.info(f"Saved LabelEncoder to {le_path}")

        logger.info("✅ Successfully generated all engineered features! ✅")
