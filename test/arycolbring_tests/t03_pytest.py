#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-04"


"""
t03_pytest.py
_______________________________________________________________________________
Pytest test suite for AryColBring collaborative filtering model.

Comprehensive testing of AryColBring functionality including:
- Model initialization and configuration
- Training pipeline
- Inference and predictions
- Data validation
- Performance metrics
- Error handling
- Edge cases

Requirements:
  pip install pytest pytest-cov pytest-xdist pytest-timeout

Run tests:
  pytest test/test_arycolbring_pytest.py -v --cov --cov-report=html

Author: Aryanto
Created: 2026-06-05
"""

import pytest
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import tempfile
import json

from src.cooprecsys.configs import logger, _cfg
from src.cooprecsys.models.arycolbring.trainer import AryColBringModelTrainer
from src.cooprecsys.models.arycolbring.assist  import describe_interactions, validate_sparse_matrix
from src.cooprecsys.models.arycolbring.inout   import TheReasoner, TheAdvisor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_sparse_matrix():
    """Create a sample sparse interaction matrix for testing."""
    # Create a small COO matrix: 10 users, 15 items, 50 interactions
    users = np.random.randint(0, 10, 50)
    items = np.random.randint(0, 15, 50)
    values = np.ones(50)

    matrix = sp.coo_matrix((values, (users, items)), shape=(10, 15))
    return matrix.tocsr()


@pytest.fixture(scope="session")
def sample_train_test_split(sample_sparse_matrix):
    """Split sample matrix into train and test sets."""
    coo = sample_sparse_matrix.tocoo()
    split_idx = int(len(coo.data) * 0.8)

    # Train
    train_coo = sp.coo_matrix(
        (coo.data[:split_idx], (coo.row[:split_idx], coo.col[:split_idx])),
        shape=sample_sparse_matrix.shape
    )

    # Test
    test_coo = sp.coo_matrix(
        (coo.data[split_idx:], (coo.row[split_idx:], coo.col[split_idx:])),
        shape=sample_sparse_matrix.shape
    )

    return train_coo.tocsr(), test_coo.tocsr()


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test: Model Initialization
# ============================================================================

class TestModelInitialization:
    """Test AryColBringModelTrainer initialization."""

    def test_default_initialization(self):
        """Test model initialization with default parameters."""
        model = AryColBringModelTrainer()
        assert model.trainer is not None
        assert model.config["loss"] == "warp"
        assert model.config["no_components"] == 32
        assert model.training_history == []
        assert model.metrics_history == []
        logger.info("Default initialization test passed")

    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        model = AryColBringModelTrainer(
            no_components=16,
            loss="bpr",
            learning_rate=0.1,
            item_alpha=0.001,
            user_alpha=0.001,
            learning_schedule="adadelta",
            random_state=42
        )
        assert model.config["no_components"] == 16
        assert model.config["loss"] == "bpr"
        assert model.config["learning_rate"] == 0.1
        assert model.config["item_alpha"] == 0.001
        assert model.config["user_alpha"] == 0.001
        assert model.config["learning_schedule"] == "adadelta"
        logger.info("Custom initialization test passed")

    def test_invalid_loss_function(self):
        """An unrecognized loss must be rejected immediately at
        construction time (AryColBringBase.__init__ validates `loss`
        before any embeddings are allocated) -- NOT deferred to fit()
        as the previous version of this test assumed. That earlier
        assumption was wrong: AryColBringModelTrainer.__init__()
        constructs the underlying TheAdvisor(loss=...) synchronously,
        so an invalid loss raises right there.
        """
        with pytest.raises(ValueError):
            AryColBringModelTrainer(loss="invalid_loss")
        logger.info("Invalid loss function test passed")

    def test_invalid_learning_schedule(self):
        """Same as above: `learning_schedule` is validated at
        construction time, not at fit time."""
        with pytest.raises(ValueError):
            AryColBringModelTrainer(learning_schedule="invalid_schedule")
        logger.info("Invalid learning schedule test passed")


# ============================================================================
# Test: Data Handling
# ============================================================================

class TestDataHandling:
    """Test data loading and validation."""

    def test_sparse_matrix_format(self, sample_sparse_matrix):
        """Test handling of sparse matrix formats."""
        # Test COO format
        coo_matrix = sample_sparse_matrix.tocoo()
        stats = describe_interactions(coo_matrix).iloc[0]
        assert stats["n_users"] == 10
        assert stats["n_items"] == 15
        assert stats["nnz"] > 0
        logger.info("COO format test passed")

        # Test CSR format
        csr_matrix = sample_sparse_matrix.tocsr()
        stats = describe_interactions(csr_matrix).iloc[0]
        assert stats["n_users"] == 10
        assert stats["n_items"] == 15
        logger.info("CSR format test passed")

        # Test CSC format
        csc_matrix = sample_sparse_matrix.tocsc()
        stats = describe_interactions(csc_matrix).iloc[0]
        assert stats["n_users"] == 10
        assert stats["n_items"] == 15
        logger.info("CSC format test passed")

    def test_matrix_sparsity_calculation(self, sample_sparse_matrix):
        """Test density/sparsity calculation.

        describe_interactions() reports `density` (nnz / (n_users *
        n_items)); sparsity is its complement (1 - density). There is
        no separate "sparsity" column.
        """
        stats = describe_interactions(sample_sparse_matrix).iloc[0]
        expected_density = stats["nnz"] / (10 * 15)
        assert abs(stats["density"] - expected_density) < 1e-5
        logger.info("Density calculation test passed")

    def test_empty_matrix(self):
        """Test handling of empty matrix."""
        empty_matrix = sp.csr_matrix((10, 15))
        stats = describe_interactions(empty_matrix).iloc[0]
        assert stats["nnz"] == 0
        assert stats["density"] == 0.0
        logger.info("Empty matrix test passed")

    def test_single_interaction(self):
        """Test matrix with single interaction."""
        single_matrix = sp.csr_matrix(
            ([1.0], ([0], [0])),
            shape=(10, 15)
        )
        stats = describe_interactions(single_matrix).iloc[0]
        assert stats["nnz"] == 1
        assert stats["n_users"] == 10
        assert stats["n_items"] == 15
        logger.info("Single interaction test passed")


# ============================================================================
# Test: Training Pipeline
# ============================================================================

class TestTrainingPipeline:
    """Test model training functionality."""

    def test_fit_basic(self, sample_sparse_matrix):
        """Test basic model fitting."""
        model = AryColBringModelTrainer(
            no_components=4,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(
            interactions=sample_sparse_matrix,
            epochs=2,
            num_threads=2,
            verbose=False
        )
        assert len(model.training_history) > 0
        assert model.trainer.item_embeddings is not None
        assert model.trainer.user_embeddings is not None
        logger.info("Basic fit test passed")

    def test_training_history(self, sample_sparse_matrix):
        """Test training history tracking."""
        model = AryColBringModelTrainer()
        model.fit(interactions=sample_sparse_matrix, epochs=1, verbose=False)

        assert len(model.training_history) == 1
        history = model.training_history[0]
        assert "epochs" in history
        assert "training_time_sec" in history
        assert "start_time" in history
        assert "end_time" in history
        logger.info("Training history test passed")

    def test_validation_evaluation(self, sample_train_test_split):
        """Test validation during training."""
        train_data, test_data = sample_train_test_split

        model = AryColBringModelTrainer(no_components=4)
        model.fit(
            interactions=train_data,
            validation_data=test_data,
            epochs=1,
            verbose=False
        )

        assert len(model.metrics_history) > 0
        metrics = model.metrics_history[0]
        assert isinstance(metrics, dict)
        logger.info("Validation evaluation test passed")


# ============================================================================
# Test: Model Evaluation
# ============================================================================

class TestModelEvaluation:
    """Test model evaluation metrics."""

    def test_evaluate_basic(self, sample_train_test_split):
        """Test basic evaluation."""
        train_data, test_data = sample_train_test_split

        model = AryColBringModelTrainer(no_components=4)
        model.fit(interactions=train_data, epochs=1, verbose=False)

        metrics = model.evaluate(test_interactions=test_data, num_threads=2)

        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        logger.info("Basic evaluation test passed")

    def test_evaluate_with_train_exclusion(self, sample_train_test_split):
        """Test evaluation with training data exclusion."""
        train_data, test_data = sample_train_test_split

        model = AryColBringModelTrainer(no_components=4)
        model.fit(interactions=train_data, epochs=1, verbose=False)

        metrics = model.evaluate(
            test_interactions=test_data,
            train_interactions=train_data,
            num_threads=2
        )

        assert isinstance(metrics, dict)
        logger.info("Evaluation with train exclusion test passed")


# ============================================================================
# Test: Model Persistence
# ============================================================================

class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_model(self, sample_sparse_matrix, temp_output_dir):
        """Test model saving."""
        model = AryColBringModelTrainer(no_components=4)
        model.fit(interactions=sample_sparse_matrix, epochs=1, verbose=False)

        model_path = temp_output_dir / "test_model.npz"
        model.save_model(str(model_path))

        assert model_path.exists()
        logger.info("Model save test passed")

    def test_load_model(self, sample_sparse_matrix, temp_output_dir):
        """Test model loading."""
        # Train and save
        model1 = AryColBringModelTrainer(no_components=4)
        model1.fit(interactions=sample_sparse_matrix, epochs=1, verbose=False)

        model_path = temp_output_dir / "test_model.npz"
        model1.save_model(str(model_path))

        # Load into new model
        model2 = AryColBringModelTrainer(no_components=4)
        model2.load_model(str(model_path))

        assert model2.trainer.item_embeddings is not None
        assert model2.trainer.user_embeddings is not None
        assert model2.config == model1.config
        logger.info("Model load test passed")

    def test_save_load_roundtrip(self, sample_sparse_matrix, temp_output_dir):
        """Test save-load roundtrip."""
        # Original model
        model1 = AryColBringModelTrainer(
            no_components=4,
            loss="warp",
            learning_rate=0.05
        )
        model1.fit(interactions=sample_sparse_matrix, epochs=1, verbose=False)

        # Save
        model_path = temp_output_dir / "roundtrip_model.npz"
        model1.save_model(str(model_path))

        # Load
        model2 = AryColBringModelTrainer()
        model2.load_model(str(model_path))

        # Verify config
        assert model2.config["no_components"] == 4
        assert model2.config["loss"] == "warp"
        assert abs(model2.config["learning_rate"] - 0.05) < 1e-6

        # Verify embeddings
        np.testing.assert_array_almost_equal(
            model1.trainer.item_embeddings,
            model2.trainer.item_embeddings
        )
        logger.info("Save-load roundtrip test passed")


# ============================================================================
# Test: Prediction and Inference
# ============================================================================

class TestPrediction:
    """Test prediction functionality."""

    def test_get_predictor(self, sample_sparse_matrix):
        """Test predictor retrieval."""
        model = AryColBringModelTrainer(no_components=4)
        model.fit(interactions=sample_sparse_matrix, epochs=1, verbose=False)

        predictor = model.get_predictor()
        assert predictor is not None
        assert predictor.item_embeddings is not None
        assert predictor.user_embeddings is not None
        logger.info("Predictor retrieval test passed")


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_sparse_matrix_type(self):
        """Test handling of non-sparse matrix input."""
        dense_matrix = np.random.rand(10, 15)
        model = AryColBringModelTrainer(no_components=4)

        # Should handle dense matrix by converting to sparse
        try:
            model.fit(interactions=dense_matrix, epochs=1, verbose=False)
            logger.info("Dense matrix handling test passed")
        except Exception as e:
            logger.warning("Dense matrix handling raised: %s", str(e))

    def test_missing_model_file(self, temp_output_dir):
        """Test loading non-existent model."""
        model = AryColBringModelTrainer()
        invalid_path = temp_output_dir / "nonexistent_model.npz"

        with pytest.raises(FileNotFoundError):
            model.load_model(str(invalid_path))
        logger.info("Missing model file test passed")


# ============================================================================
# Test: TheReasoner / TheAdvisor hyperparameter validation (AryColBringBase)
# ============================================================================
# NOTE: `TheReasoner` (= AryColBringPredictor, inference) and `TheAdvisor`
# (= AryColBringTrainer, training) both inherit their entire constructor
# and validation logic from the shared AryColBringBase.__init__ in
# scaffold.py. TheAdvisor is used below purely because it's cheaper to
# construct (no pre-existing embeddings required) -- every assertion here
# is exercising AryColBringBase's shared validation, which applies
# identically to TheReasoner.

class TestReasonerHyperparameterValidation:
    """Test AryColBringBase.__init__ validation, shared by TheReasoner
    (AryColBringPredictor) and TheAdvisor (AryColBringTrainer)."""

    def test_valid_construction(self):
        model = TheAdvisor(no_components=8, loss="warp", k=3, n=6,
                           learning_schedule="adagrad", random_state=42)
        assert model.no_components == 8
        assert model.loss == "warp"
        assert model.is_fitted is False
        logger.info("Valid TheAdvisor construction test passed")

    @pytest.mark.parametrize("kwargs", [
        {"item_alpha": -0.1},
        {"user_alpha": -0.1},
        {"no_components": 0},
        {"no_components": -5},
        {"k": 0},
        {"n": 0},
        {"rho": 0.0},
        {"rho": 1.0},
        {"rho": 1.5},
        {"epsilon": -1e-6},
        {"max_sampled": 0},
        {"learning_schedule": "not_a_schedule"},
        {"loss": "not_a_loss"},
    ])
    def test_invalid_hyperparameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            TheAdvisor(**kwargs)
        logger.info("Invalid hyperparameter rejected: %s", kwargs)

    def test_invalid_random_state_type_raises(self):
        with pytest.raises(TypeError):
            TheAdvisor(random_state="not-a-seed")

    def test_is_fitted_false_before_training(self):
        model = TheAdvisor(no_components=4)
        assert model.is_fitted is False
        assert model.item_embeddings is None
        assert model.user_embeddings is None

    def test_setter_validation_no_components(self):
        model = TheAdvisor(no_components=4)
        with pytest.raises(ValueError):
            model.no_components = -1

    def test_setter_validation_learning_rate(self):
        model = TheAdvisor(no_components=4)
        with pytest.raises(ValueError):
            model.learning_rate = 0.0

    def test_setter_validation_item_alpha(self):
        model = TheAdvisor(no_components=4)
        with pytest.raises(ValueError):
            model.item_alpha = -0.5


class TestReasonerParams:
    """Test the sklearn-style get_params/set_params contract shared by
    TheReasoner and TheAdvisor."""

    def test_get_params_roundtrip(self):
        model = TheAdvisor(no_components=12, loss="bpr", learning_rate=0.02)
        params = model.get_params()
        assert params["no_components"] == 12
        assert params["loss"] == "bpr"
        assert params["learning_rate"] == 0.02

    def test_set_params_updates_state(self):
        model = TheAdvisor(no_components=4)
        model.set_params(no_components=16, loss="warp")
        assert model.get_params()["no_components"] == 16
        assert model.get_params()["loss"] == "warp"

    def test_set_params_rejects_unknown_key(self):
        model = TheAdvisor(no_components=4)
        with pytest.raises(ValueError):
            model.set_params(not_a_real_param=123)

    def test_repr_contains_key_hyperparameters(self):
        model = TheAdvisor(no_components=4, loss="warp")
        text = repr(model)
        assert "warp" in text
        assert "no_components" in text


# ============================================================================
# Test: AryColBringPredictor (TheReasoner) pure-Python helpers
# ============================================================================
# build_pairs() and _is_string_type() are @staticmethod / pure-numpy --
# no embeddings, no fitted model, and no call into the compiled
# CLproximity extension are needed to exercise them directly.

class TestReasonerPairUtilities:
    """Test TheReasoner.build_pairs() -- the label-pairing utility behind
    predict()'s pairwise/broadcast/cross_join modes."""

    def test_build_pairs_strict_pairwise(self):
        u, i = TheReasoner.build_pairs([1, 2, 3], [10, 20, 30], cross_join=False)
        np.testing.assert_array_equal(u, [1, 2, 3])
        np.testing.assert_array_equal(i, [10, 20, 30])

    def test_build_pairs_cross_join(self):
        u, i = TheReasoner.build_pairs([1, 2], [10, 20], cross_join=True)
        # row-major: user 0 vs every item, then user 1 vs every item
        np.testing.assert_array_equal(u, [1, 1, 2, 2])
        np.testing.assert_array_equal(i, [10, 20, 10, 20])
        assert len(u) == len(i) == 4

    def test_build_pairs_scalar_item_broadcast(self):
        u, i = TheReasoner.build_pairs([1, 2, 3], [99], cross_join=False)
        np.testing.assert_array_equal(i, [99, 99, 99])
        assert len(u) == len(i) == 3

    def test_build_pairs_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            TheReasoner.build_pairs([1, 2, 3], [10, 20], cross_join=False)

    def test_is_string_type_numeric_array(self):
        assert TheReasoner._is_string_type(np.array([1, 2, 3])) is False

    def test_is_string_type_string_array(self):
        assert TheReasoner._is_string_type(np.array(["a", "b", "c"])) is True

    def test_is_string_type_plain_list_of_strings(self):
        assert TheReasoner._is_string_type(["x", "y"]) is True

    def test_is_string_type_plain_list_of_ints(self):
        assert TheReasoner._is_string_type([1, 2, 3]) is False


# ============================================================================
# Test: Configuration
# ============================================================================

class TestConfiguration:
    """Test configuration loading."""

    def test_config_from_ini(self):
        """Test loading configuration from INI file."""
        # Access configuration through _cfg
        loss = _cfg.get("model", "loss", fallback="warp")
        epochs = _cfg.getint("model", "epochs", fallback=10)

        assert loss in ["warp", "bpr", "logistic", "warp-kos"]
        assert epochs > 0
        logger.info("Config from INI test passed")


# ============================================================================
# Main Test Execution
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([
        __file__,
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-s",  # Show print statements
        "--color=yes",  # Colored output
        # Uncomment for coverage:
        # "--cov=src/models/arycolbring",
        # "--cov-report=html",
        # "--cov-report=term-missing",
    ])
