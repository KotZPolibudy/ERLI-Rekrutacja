"""Klasyfikatory kategorii produktów:
1. SigLIPClassifier - Vision-Language z sigmoid loss (natywny multi-label)
2. CLIPClassifier - Vision-Language z softmax loss (klasyczny baseline)
3. LinearProbeClassifier - głowica wieloetykietowa na zamrożonych embeddingach SigLIP
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoModel, AutoProcessor

from src.classification.prompts import get_category_prompts
from src.common.config import MAIN_CATEGORIES
from src.common.schemas import CategoryPrediction, CategoryScore

logger = logging.getLogger("classification_models")


class BaseClassifier(ABC):
    """Abstrakcyjna klasa bazowa dla klasyfikatorów kategorii Erli."""

    @abstractmethod
    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image],
        product_id: str = "unknown",
        true_category: Optional[str] = None,
        threshold: float = 0.35,
    ) -> CategoryPrediction:
        """Klasyfikuje pojedynczy obraz i zwraca obiekt CategoryPrediction."""
        pass

    @abstractmethod
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        product_ids: Optional[List[str]] = None,
        true_categories: Optional[List[str]] = None,
        threshold: float = 0.35,
    ) -> List[CategoryPrediction]:
        """Klasyfikuje paczkę obrazów z zachowaniem optymalizacji batchingu."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Zwraca identyfikator modelu."""
        pass


def _extract_tensor(features: Any) -> torch.Tensor:
    """Ekstrahuje właściwy tensor z wyjścia get_text_features / get_image_features (np. BaseModelOutputWithPooling)."""
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0]
    if isinstance(features, (list, tuple)):
        return features[0]
    return features


class SigLIPClassifier(BaseClassifier):
    """Klasyfikator Vision-Language oparty o Google SigLIP (Sigmoid Loss for Language-Image Pre-training).
    Zaprojektowany z myślą o multi-label – prawdopodobieństwa nie sumują się sztucznie do 1.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        prompt_lang: str = "en",
        use_descriptions: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Ładowanie modelu SigLIP (%s) na urządzenie %s...", self.model_name, self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.categories = MAIN_CATEGORIES
        self.prompts = get_category_prompts(lang=prompt_lang, use_descriptions=use_descriptions)

        # Prekomputacja embeddingów tekstowych dla 12 kategorii (oszczędza czas przy inferencji)
        self._precompute_text_embeddings()

    def _precompute_text_embeddings(self) -> None:
        """Jednorazowo oblicza znormalizowane wektory cech dla promptów tekstowych."""
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.prompts,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            ).to(self.device)
            text_features = _extract_tensor(self.model.get_text_features(**text_inputs))
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logger.info("Wyznaczono embeddingi tekstowe dla %d kategorii.", len(self.prompts))

    def get_model_name(self) -> str:
        return f"SigLIP ({self.model_name})"

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image],
        product_id: str = "unknown",
        true_category: Optional[str] = None,
        threshold: float = 0.35,
    ) -> CategoryPrediction:
        batch_res = self.predict_batch(
            images=[image_input],
            product_ids=[product_id],
            true_categories=[true_category] if true_category else None,
            threshold=threshold,
        )
        return batch_res[0]

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        product_ids: Optional[List[str]] = None,
        true_categories: Optional[List[str]] = None,
        threshold: float = 0.35,
    ) -> List[CategoryPrediction]:
        t0 = time.perf_counter()
        loaded_images: List[Image.Image] = []
        paths: List[str] = []

        for img in images:
            if isinstance(img, (str, Path)):
                paths.append(str(img))
                with Image.open(img) as pil_img:
                    loaded_images.append(pil_img.convert("RGB"))
            else:
                paths.append("in_memory")
                loaded_images.append(img.convert("RGB"))

        with torch.no_grad():
            image_inputs = self.processor(images=loaded_images, return_tensors="pt").to(self.device)
            image_features = _extract_tensor(self.model.get_image_features(**image_inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Obliczenie logitów (podobieństwo cosinusowe ze skalą SigLIP, bez marginalnego WebLI biasu)
            logit_scale = torch.clamp(self.model.logit_scale.exp(), max=100.0)
            unbiased_logits = (image_features @ self.text_features.T) * logit_scale
            # SigLIP używa SIGMOID dla niezależnych prawdopodobieństw wieloetykietowych
            probabilities = torch.sigmoid(unbiased_logits * 0.5).cpu().numpy()

        elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_latency_ms = elapsed_total_ms / max(1, len(images))

        results: List[CategoryPrediction] = []
        for i, probs in enumerate(probabilities):
            p_id = product_ids[i] if product_ids else f"item_{i}"
            true_cat = true_categories[i] if true_categories else None
            img_path = paths[i]

            # Posortuj kategorie malejąco po prawdopodobieństwie
            sorted_indices = np.argsort(probs)[::-1]
            top_k = [
                CategoryScore(category=self.categories[idx], score=float(probs[idx]))
                for idx in sorted_indices
            ]

            # Najbardziej prawdopodobna kategoria (single-label)
            best_cat = top_k[0].category
            best_conf = top_k[0].score

            # Wieloetykietowość (multi-label) – wszystkie kategorie powyżej progu threshold
            multilabel_cats = [
                self.categories[idx]
                for idx in sorted_indices
                if probs[idx] >= threshold
            ]
            if not multilabel_cats:
                # Jeśli żadna nie przekracza progu, weź najwyższą
                multilabel_cats = [best_cat]

            results.append(CategoryPrediction(
                product_id=p_id,
                image_path=img_path,
                true_category=true_cat,
                true_categories=[true_cat] if true_cat else [],
                predicted_category=best_cat,
                predicted_categories=multilabel_cats,
                is_multilabel=(len(multilabel_cats) > 1),
                confidence=best_conf,
                top_k_predictions=top_k,
                latency_ms=per_item_latency_ms,
            ))

        return results


class CLIPClassifier(BaseClassifier):
    """Klasyfikator Vision-Language oparty o OpenAI CLIP (Softmax loss baseline)."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        prompt_lang: str = "en",
        use_descriptions: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Ładowanie modelu CLIP (%s) na urządzenie %s...", self.model_name, self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.categories = MAIN_CATEGORIES
        self.prompts = get_category_prompts(lang=prompt_lang, use_descriptions=use_descriptions)
        self._precompute_text_embeddings()

    def _precompute_text_embeddings(self) -> None:
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.prompts,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            text_features = _extract_tensor(self.model.get_text_features(**text_inputs))
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def get_model_name(self) -> str:
        return f"CLIP ({self.model_name})"

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image],
        product_id: str = "unknown",
        true_category: Optional[str] = None,
        threshold: float = 0.20,
    ) -> CategoryPrediction:
        batch_res = self.predict_batch(
            images=[image_input],
            product_ids=[product_id],
            true_categories=[true_category] if true_category else None,
            threshold=threshold,
        )
        return batch_res[0]

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        product_ids: Optional[List[str]] = None,
        true_categories: Optional[List[str]] = None,
        threshold: float = 0.20,
    ) -> List[CategoryPrediction]:
        t0 = time.perf_counter()
        loaded_images = []
        paths = []

        for img in images:
            if isinstance(img, (str, Path)):
                paths.append(str(img))
                with Image.open(img) as pil_img:
                    loaded_images.append(pil_img.convert("RGB"))
            else:
                paths.append("in_memory")
                loaded_images.append(img.convert("RGB"))

        with torch.no_grad():
            image_inputs = self.processor(images=loaded_images, return_tensors="pt").to(self.device)
            image_features = _extract_tensor(self.model.get_image_features(**image_inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # CLIP używa SOFTMAX
            logits = (image_features @ self.text_features.T) * self.model.logit_scale.exp()
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_latency_ms = elapsed_total_ms / max(1, len(images))

        results: List[CategoryPrediction] = []
        for i, probs in enumerate(probabilities):
            p_id = product_ids[i] if product_ids else f"item_{i}"
            true_cat = true_categories[i] if true_categories else None
            img_path = paths[i]

            sorted_indices = np.argsort(probs)[::-1]
            top_k = [
                CategoryScore(category=self.categories[idx], score=float(probs[idx]))
                for idx in sorted_indices
            ]
            best_cat = top_k[0].category
            best_conf = top_k[0].score

            multilabel_cats = [
                self.categories[idx]
                for idx in sorted_indices
                if probs[idx] >= threshold
            ] or [best_cat]

            results.append(CategoryPrediction(
                product_id=p_id,
                image_path=img_path,
                true_category=true_cat,
                true_categories=[true_cat] if true_cat else [],
                predicted_category=best_cat,
                predicted_categories=multilabel_cats,
                is_multilabel=(len(multilabel_cats) > 1),
                confidence=best_conf,
                top_k_predictions=top_k,
                latency_ms=per_item_latency_ms,
            ))

        return results


class LinearProbeClassifier(BaseClassifier):
    """Supervised Linear Probe trenowany na zamrożonych embeddingach SigLIP."""

    def __init__(self, feature_extractor: Optional[SigLIPClassifier] = None):
        self.feature_extractor = feature_extractor or SigLIPClassifier()
        # One-vs-Rest Logistic Regression dla wsparcia Multi-Label
        self.clf = OneVsRestClassifier(LogisticRegression(max_iter=500, C=1.0))
        self.categories = MAIN_CATEGORIES
        self.is_fitted = False

    def get_model_name(self) -> str:
        return "LinearProbe (SigLIP Features + Multi-Label Logistic Regression)"

    def fit(self, images: List[Union[str, Path, Image.Image]], targets: np.ndarray) -> None:
        """Uczy głowicę klasyfikacyjną na wektorach cech i macierzy etykiet (multi-hot lub one-hot)."""
        logger.info("Ekstrakcja embeddingów dla %d próbek treningowych...", len(images))
        X = self._extract_features(images)
        logger.info("Trenowanie klasyfikatora One-vs-Rest...")
        self.clf.fit(X, targets)
        self.is_fitted = True
        logger.info("Linear Probe został pomyślnie wytrenowany.")

    def _extract_features(self, images: List[Union[str, Path, Image.Image]]) -> np.ndarray:
        loaded = []
        for img in images:
            if isinstance(img, (str, Path)):
                with Image.open(img) as pil_img:
                    loaded.append(pil_img.convert("RGB"))
            else:
                loaded.append(img.convert("RGB"))

        with torch.no_grad():
            inputs = self.feature_extractor.processor(images=loaded, return_tensors="pt").to(self.feature_extractor.device)
            feats = _extract_tensor(self.feature_extractor.model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image],
        product_id: str = "unknown",
        true_category: Optional[str] = None,
        threshold: float = 0.35,
    ) -> CategoryPrediction:
        return self.predict_batch([image_input], [product_id], [true_category] if true_category else None, threshold)[0]

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        product_ids: Optional[List[str]] = None,
        true_categories: Optional[List[str]] = None,
        threshold: float = 0.35,
    ) -> List[CategoryPrediction]:
        if not self.is_fitted:
            raise RuntimeError("LinearProbeClassifier musi zostać najpierw wytrenowany metodą fit()!")

        t0 = time.perf_counter()
        feats = self._extract_features(images)
        probabilities = self.clf.predict_proba(feats)
        elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_latency_ms = elapsed_total_ms / max(1, len(images))

        results = []
        for i, probs in enumerate(probabilities):
            p_id = product_ids[i] if product_ids else f"item_{i}"
            true_cat = true_categories[i] if true_categories else None
            path_str = str(images[i]) if isinstance(images[i], (str, Path)) else "in_memory"

            sorted_indices = np.argsort(probs)[::-1]
            top_k = [
                CategoryScore(category=self.categories[idx], score=float(probs[idx]))
                for idx in sorted_indices
            ]
            best_cat = top_k[0].category
            best_conf = top_k[0].score

            multilabel_cats = [
                self.categories[idx]
                for idx in sorted_indices
                if probs[idx] >= threshold
            ] or [best_cat]

            results.append(CategoryPrediction(
                product_id=p_id,
                image_path=path_str,
                true_category=true_cat,
                true_categories=[true_cat] if true_cat else [],
                predicted_category=best_cat,
                predicted_categories=multilabel_cats,
                is_multilabel=(len(multilabel_cats) > 1),
                confidence=best_conf,
                top_k_predictions=top_k,
                latency_ms=per_item_latency_ms,
            ))

        return results
