"""
Semantic drift detection for LLM outputs.

This module provides drift detection based on semantic content
analysis, including perplexity, topic modeling, and coherence metrics.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import os
from collections import Counter

from .base import BaseDriftDetector, DriftResult
from ..data.processor import ProcessedSample
from ..config import Config

logger = logging.getLogger(__name__)

class SemanticDriftDetector(BaseDriftDetector):
    """
    Detects drift based on semantic content analysis.
    
    This detector analyzes the semantic content of LLM outputs to detect
    changes in topics, coherence, writing style, and other semantic properties.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the semantic drift detector.
        
        Args:
            config: Configuration object, uses global config if None
        """
        super().__init__(config)
        
        # Get semantic analysis method
        self.method = self.detector_config.get("method", "perplexity")
        
        # Flag to track model loading
        self._model_loaded = False
        self._perplexity_model = None
        self._tokenizer = None
        
        # Load models if required
        if self.enabled and self.method == "perplexity":
            self._load_perplexity_model()
    
    def _load_perplexity_model(self):
        """
        Load model for perplexity calculation.
        
        This will conditionally import and load the model only when needed.
        """
        if self._model_loaded:
            return
        
        model_name = self.detector_config.get("model", "distilgpt2")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading perplexity model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self._perplexity_model = self._perplexity_model.cuda()
            
            self._model_loaded = True
            logger.info(f"Perplexity model loaded successfully")
            
        except ImportError:
            logger.warning("transformers or torch not installed. Perplexity calculation will be disabled.")
            self._model_loaded = False
        except Exception as e:
            logger.warning(f"Error loading perplexity model: {str(e)}")
            self._model_loaded = False
    
    def detect(self, 
              reference_samples: List[ProcessedSample],
              current_samples: List[ProcessedSample]) -> DriftResult:
        """
        Detect semantic drift between reference and current samples.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            DriftResult: Result of drift detection
        """
        # Validate inputs
        if not self._validate_input(reference_samples, current_samples):
            return self._create_result(
                0.0, reference_samples, current_samples,
                analysis={"error": "Invalid input or detector disabled"}
            )
        
        # Dispatch to appropriate method
        if self.method == "perplexity":
            return self._detect_perplexity_drift(reference_samples, current_samples)
        elif self.method == "topic":
            return self._detect_topic_drift(reference_samples, current_samples)
        elif self.method == "lexical":
            return self._detect_lexical_drift(reference_samples, current_samples)
        else:
            # Default to lexical drift detection (doesn't require additional dependencies)
            return self._detect_lexical_drift(reference_samples, current_samples)
    
    def _detect_perplexity_drift(self, 
                               reference_samples: List[ProcessedSample],
                               current_samples: List[ProcessedSample]) -> DriftResult:
        """
        Detect drift based on perplexity distribution.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            DriftResult: Result of drift detection
        """
        # Check if perplexity model is loaded
        if not self._model_loaded:
            # Fall back to lexical drift detection
            logger.warning("Perplexity model not loaded. Falling back to lexical drift detection.")
            return self._detect_lexical_drift(reference_samples, current_samples)
        
        # Calculate perplexity for reference samples
        ref_perplexities = self._calculate_perplexities([s.response for s in reference_samples])
        
        # Calculate perplexity for current samples
        curr_perplexities = self._calculate_perplexities([s.response for s in current_samples])
        
        # Calculate perplexity statistics
        ref_mean = np.mean(ref_perplexities)
        curr_mean = np.mean(curr_perplexities)
        
        # Calculate relative change in perplexity
        rel_change = abs(curr_mean - ref_mean) / max(ref_mean, 1e-10)
        
        # Scale to a 0-1 range for drift score
        perplexity_threshold = self.detector_config.get("perplexity_threshold", 0.3)
        drift_score = min(rel_change / perplexity_threshold, 1.0)
        
        # Prepare analysis results
        analysis = {
            "reference_perplexity": {
                "mean": float(ref_mean),
                "std": float(np.std(ref_perplexities)),
                "min": float(np.min(ref_perplexities)),
                "max": float(np.max(ref_perplexities))
            },
            "current_perplexity": {
                "mean": float(curr_mean),
                "std": float(np.std(curr_perplexities)),
                "min": float(np.min(curr_perplexities)),
                "max": float(np.max(curr_perplexities))
            },
            "relative_change": float(rel_change),
            "method": "perplexity"
        }
        
        return self._create_result(
            drift_score, reference_samples, current_samples,
            metric_name="perplexity_drift",
            analysis=analysis
        )
    
    def _calculate_perplexities(self, texts: List[str]) -> np.ndarray:
        """
        Calculate perplexity for a list of texts.
        
        Args:
            texts: List of text samples
            
        Returns:
            np.ndarray: Array of perplexity values
        """
        import torch
        
        perplexities = []
        max_length = self._tokenizer.model_max_length
        
        with torch.no_grad():
            for text in texts:
                # Truncate to avoid excessive computation
                if len(text) > 1000:
                    text = text[:1000]
                
                # Tokenize
                encodings = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                
                # Move to GPU if model is on GPU
                if next(self._perplexity_model.parameters()).is_cuda:
                    encodings = {k: v.cuda() for k, v in encodings.items()}
                
                # Calculate loss
                outputs = self._perplexity_model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss.item()
                
                # Calculate perplexity
                perplexity = np.exp(loss)
                perplexities.append(perplexity)
        
        return np.array(perplexities)
    
    def _detect_topic_drift(self, 
                          reference_samples: List[ProcessedSample],
                          current_samples: List[ProcessedSample]) -> DriftResult:
        """
        Detect drift based on topic distribution.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            DriftResult: Result of drift detection
        """
        try:
            # Try to import BERTopic
            from bertopic import BERTopic
            import umap
            
            # Extract responses
            ref_texts = [s.response for s in reference_samples]
            curr_texts = [s.response for s in current_samples]
            
            # Create and fit BERTopic model
            topic_model = BERTopic(verbose=False)
            
            # Fit on reference data
            _, reference_topics = topic_model.fit_transform(ref_texts)
            
            # Transform current data
            _, current_topics = topic_model.transform(curr_texts)
            
            # Calculate topic distributions
            ref_topic_dist = np.bincount(reference_topics + 1, minlength=max(reference_topics + 1) + 1)
            curr_topic_dist = np.bincount(current_topics + 1, minlength=max(current_topics + 1) + 1)
            
            # Normalize
            ref_topic_dist = ref_topic_dist / np.sum(ref_topic_dist)
            curr_topic_dist = curr_topic_dist / np.sum(curr_topic_dist)
            
            # Ensure same length
            max_len = max(len(ref_topic_dist), len(curr_topic_dist))
            if len(ref_topic_dist) < max_len:
                ref_topic_dist = np.pad(ref_topic_dist, (0, max_len - len(ref_topic_dist)))
            if len(curr_topic_dist) < max_len:
                curr_topic_dist = np.pad(curr_topic_dist, (0, max_len - len(curr_topic_dist)))
            
            # Calculate Jensen-Shannon divergence
            m = (ref_topic_dist + curr_topic_dist) / 2
            js_div = 0.5 * np.sum(ref_topic_dist * np.log(ref_topic_dist / m + 1e-10)) + \
                    0.5 * np.sum(curr_topic_dist * np.log(curr_topic_dist / m + 1e-10))
            
            # Scale to a 0-1 range for drift score
            topic_threshold = self.detector_config.get("topic_drift_threshold", 0.4)
            drift_score = min(js_div / topic_threshold, 1.0)
            
            # Prepare analysis results
            analysis = {
                "jensen_shannon_divergence": float(js_div),
                "reference_topic_count": int(np.count_nonzero(ref_topic_dist)),
                "current_topic_count": int(np.count_nonzero(curr_topic_dist)),
                "method": "topic_modeling"
            }
            
            return self._create_result(
                drift_score, reference_samples, current_samples,
                metric_name="topic_drift",
                analysis=analysis
            )
            
        except ImportError:
            logger.warning("BERTopic or dependencies not installed. Falling back to lexical drift detection.")
            return self._detect_lexical_drift(reference_samples, current_samples)
        except Exception as e:
            logger.warning(f"Error in topic drift detection: {str(e)}. Falling back to lexical drift detection.")
            return self._detect_lexical_drift(reference_samples, current_samples)
    
    def _detect_lexical_drift(self, 
                            reference_samples: List[ProcessedSample],
                            current_samples: List[ProcessedSample]) -> DriftResult:
        """
        Detect drift based on lexical features (vocabulary, n-grams).
        
        This method doesn't require additional dependencies and can be used as a fallback.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            DriftResult: Result of drift detection
        """
        # Extract responses
        ref_texts = [s.response for s in reference_samples]
        curr_texts = [s.response for s in current_samples]
        
        # Calculate lexical features
        ref_features = self._extract_lexical_features(ref_texts)
        curr_features = self._extract_lexical_features(curr_texts)
        
        # Calculate feature differences
        vocab_diff = self._calculate_vocabulary_difference(ref_features["vocab"], curr_features["vocab"])
        style_diff = self._calculate_style_difference(ref_features, curr_features)
        
        # Calculate overall drift score (weighted average)
        drift_score = 0.6 * vocab_diff + 0.4 * style_diff
        
        # Prepare analysis results
        analysis = {
            "vocabulary_difference": float(vocab_diff),
            "style_difference": float(style_diff),
            "reference_vocab_size": len(ref_features["vocab"]),
            "current_vocab_size": len(curr_features["vocab"]),
            "reference_avg_length": float(ref_features["avg_length"]),
            "current_avg_length": float(curr_features["avg_length"]),
            "method": "lexical"
        }
        
        return self._create_result(
            drift_score, reference_samples, current_samples,
            metric_name="lexical_drift",
            analysis=analysis
        )
    
    def _extract_lexical_features(self, texts: List[str]) -> Dict[str, Any]:
        """
        Extract lexical features from a list of texts.
        
        Args:
            texts: List of text samples
            
        Returns:
            Dict[str, Any]: Dictionary of lexical features
        """
        # Tokenize texts (simple whitespace tokenization)
        tokens = []
        for text in texts:
            tokens.extend(text.lower().split())
        
        # Calculate vocabulary
        vocab = Counter(tokens)
        
        # Calculate average sentence length
        sentence_lengths = []
        for text in texts:
            sentences = text.split('.')
            for s in sentences:
                if s.strip():
                    sentence_lengths.append(len(s.split()))
        
        avg_length = np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Calculate part-of-speech ratios (approximation using simple heuristics)
        word_lengths = [len(token) for token in tokens]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        return {
            "vocab": vocab,
            "avg_length": avg_length,
            "avg_word_length": avg_word_length,
            "token_count": len(tokens),
            "unique_token_count": len(vocab)
        }
    
    def _calculate_vocabulary_difference(self, ref_vocab: Counter, curr_vocab: Counter) -> float:
        """
        Calculate vocabulary difference between reference and current texts.
        
        Args:
            ref_vocab: Counter of reference vocabulary
            curr_vocab: Counter of current vocabulary
            
        Returns:
            float: Vocabulary difference score (0-1)
        """
        # Calculate Jaccard distance between vocabularies
        ref_words = set(ref_vocab.keys())
        curr_words = set(curr_vocab.keys())
        
        intersection = ref_words.intersection(curr_words)
        union = ref_words.union(curr_words)
        
        if not union:
            return 0.0
        
        jaccard_distance = 1 - len(intersection) / len(union)
        
        # Calculate frequency distribution difference
        common_words = list(intersection)
        if not common_words:
            return jaccard_distance
        
        # Normalize frequencies
        ref_total = sum(ref_vocab.values())
        curr_total = sum(curr_vocab.values())
        
        ref_freqs = np.array([ref_vocab[w] / ref_total for w in common_words])
        curr_freqs = np.array([curr_vocab[w] / curr_total for w in common_words])
        
        # Calculate Jensen-Shannon divergence
        m = (ref_freqs + curr_freqs) / 2
        js_div = 0.5 * np.sum(ref_freqs * np.log(ref_freqs / m + 1e-10)) + \
                0.5 * np.sum(curr_freqs * np.log(curr_freqs / m + 1e-10))
        
        # Combine Jaccard distance and JS divergence
        return 0.5 * jaccard_distance + 0.5 * min(js_div, 1.0)
    
    def _calculate_style_difference(self, ref_features: Dict[str, Any], curr_features: Dict[str, Any]) -> float:
        """
        Calculate writing style difference.
        
        Args:
            ref_features: Reference lexical features
            curr_features: Current lexical features
            
        Returns:
            float: Style difference score (0-1)
        """
        # Compare average sentence length
        length_diff = abs(ref_features["avg_length"] - curr_features["avg_length"])
        max_length = max(ref_features["avg_length"], curr_features["avg_length"])
        norm_length_diff = length_diff / max_length if max_length > 0 else 0
        
        # Compare average word length
        word_length_diff = abs(ref_features["avg_word_length"] - curr_features["avg_word_length"])
        max_word_length = max(ref_features["avg_word_length"], curr_features["avg_word_length"])
        norm_word_diff = word_length_diff / max_word_length if max_word_length > 0 else 0
        
        # Compare lexical diversity
        ref_diversity = ref_features["unique_token_count"] / max(ref_features["token_count"], 1)
        curr_diversity = curr_features["unique_token_count"] / max(curr_features["token_count"], 1)
        diversity_diff = abs(ref_diversity - curr_diversity)
        
        # Combine differences
        return (0.4 * norm_length_diff + 0.3 * norm_word_diff + 0.3 * diversity_diff)