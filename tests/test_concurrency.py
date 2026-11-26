"""Deterministic Concurrency and Lifecycle Tests for ModelManager.

Verifies:
1. Active lease protection: Models actively leased cannot be evicted.
2. Simultaneous requests switching between BART and T5.
3. Condition variable unblocking upon lease release.
4. Concurrent leases on the same model with accurate reference counting.
5. Timeout safety when all cached models are actively leased.
"""

import time
import logging
import threading
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock, patch
import pytest
import torch

from app import ModelManager


class MockSummarizer:
    """Mock summarizer for deterministic concurrency testing."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.init_time_sec = 0.05
        self.cold_latency_sec = 0.1
        self.warm_latencies = []
        self.inference_count = 0
        self.is_unloaded = False
        self._inference_lock = threading.RLock()

    def warmup(self, sample_text: Optional[str] = None) -> float:
        return 0.05

    @property
    def is_inferring(self) -> bool:
        if not hasattr(self, "_inference_lock"):
            return False
        acquired = self._inference_lock.acquire(blocking=False)
        if acquired:
            self._inference_lock.release()
            return False
        return True

    def summarize_text(self, text: str, **kwargs):
        with self._inference_lock:
            if self.is_unloaded:
                raise RuntimeError(f"Attempted to run inference on unloaded model {self.model_name}!")
            time.sleep(0.05)  # Simulate forward pass
            return {
                "summary": f"Summary of {text} by {self.model_name}",
                "latency_sec": 0.05,
                "total_e2e_latency_sec": 0.06,
                "input_tokens": 15,
                "output_tokens": 10,
                "device": "cpu",
                "is_cold": False,
            }

    def summarize_batch(self, texts, batch_size=1, **kwargs):
        with self._inference_lock:
            if self.is_unloaded:
                raise RuntimeError(f"Attempted to run batch inference on unloaded model {self.model_name}!")
            return [self.summarize_text(t, **kwargs) for t in texts]

    def unload_model(self):
        with self._inference_lock:
            self.is_unloaded = True


def test_model_manager_lease_blocks_eviction():
    """Verify that an actively leased model is protected from eviction until release."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        eviction_attempted_while_leased = False
        thread1_started = threading.Event()
        thread1_finished = threading.Event()
        results = []

        def worker1_holds_bart():
            nonlocal eviction_attempted_while_leased
            with mgr.lease_model("bart") as (model, meta):
                assert mgr.is_leased("bart")
                assert mgr.get_lease_count("bart") == 1
                thread1_started.set()
                # Hold model lease for 0.25 seconds
                time.sleep(0.25)
                # Ensure model was not unloaded during inference
                assert not model.is_unloaded
                res = model.summarize_text("First article")
                results.append(("worker1", res["summary"]))
            thread1_finished.set()

        def worker2_requests_t5():
            # Wait until worker1 is inside its lease
            thread1_started.wait(timeout=2.0)
            assert mgr.is_leased("bart")
            # This should block until worker1 completes because max_cached_models=1
            t_start = time.perf_counter()
            with mgr.lease_model("t5") as (model, meta):
                t_waited = time.perf_counter() - t_start
                # Should have waited for worker1 to finish
                assert t_waited >= 0.15
                assert thread1_finished.is_set()
                assert not model.is_unloaded
                res = model.summarize_text("Second article")
                results.append(("worker2", res["summary"]))

        t1 = threading.Thread(target=worker1_holds_bart)
        t2 = threading.Thread(target=worker2_requests_t5)

        t1.start()
        t2.start()

        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        assert len(results) == 2
        assert results[0][0] == "worker1"
        assert results[1][0] == "worker2"
        assert not mgr.is_leased("bart")
        assert not mgr.is_leased("t5")


def test_model_manager_concurrent_same_model_leases():
    """Verify multiple concurrent threads can lease the same model safely."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        max_seen_leases = 0
        lock = threading.Lock()

        def worker():
            nonlocal max_seen_leases
            with mgr.lease_model("bart") as (model, meta):
                current_count = mgr.get_lease_count("bart")
                with lock:
                    if current_count > max_seen_leases:
                        max_seen_leases = current_count
                time.sleep(0.1)
                assert not model.is_unloaded

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        assert max_seen_leases >= 2, f"Expected concurrent leases, saw max {max_seen_leases}"
        assert mgr.get_lease_count("bart") == 0


def test_model_manager_timeout_when_all_leased():
    """Verify TimeoutError is raised if all cached models remain leased beyond timeout."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        lease_acquired = threading.Event()

        def blocker():
            with mgr.lease_model("bart"):
                lease_acquired.set()
                time.sleep(0.4)

        t = threading.Thread(target=blocker)
        t.start()

        lease_acquired.wait(timeout=1.0)
        # Attempting to load t5 with short timeout of 0.1s must raise TimeoutError
        with pytest.raises(TimeoutError):
            with mgr.lease_model("t5", timeout=0.1):
                pass

        t.join(timeout=2.0)
        assert mgr.get_lease_count("bart") == 0


def test_simultaneous_requests_alternating_models():
    """Stress test with 6 concurrent threads randomly alternating between BART and T5."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        errors = []

        def alternating_worker(idx: int):
            model_key = "bart" if idx % 2 == 0 else "t5"
            try:
                for loop in range(2):
                    with mgr.lease_model(model_key, timeout=5.0) as (model, meta):
                        res = model.summarize_text(f"Thread {idx} loop {loop}")
                        assert "Summary" in res["summary"]
                        assert not model.is_unloaded
            except Exception as e:
                errors.append(f"Worker {idx} error: {e}")

        threads = [threading.Thread(target=alternating_worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"


# ==============================================================================
# Priority 1 & 2 Regression Tests: Legacy Safety & Single vs Batch Concurrency
# ==============================================================================

def test_legacy_get_model_with_metadata_timeout():
    """Verify legacy get_model_with_metadata cannot evict an actively leased model."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        lease_started = threading.Event()
        lease_release = threading.Event()

        def lease_holder():
            with mgr.lease_model("bart"):
                lease_started.set()
                lease_release.wait(timeout=2.0)

        t = threading.Thread(target=lease_holder)
        t.start()
        lease_started.wait(timeout=1.0)

        # Legacy get_model_with_metadata must NOT evict bart; must raise TimeoutError if timeout expires
        with pytest.raises(TimeoutError):
            mgr.get_model_with_metadata("t5", timeout=0.1)

        assert mgr.is_leased("bart")
        lease_release.set()
        t.join(timeout=2.0)
        assert not mgr.is_leased("bart")


def test_concurrent_single_vs_batch_inference_synchronization():
    """Verify that concurrent single and batch inference synchronize safely on _inference_lock."""
    from src.summarizer import AbstractiveSummarizer
    # Test with real T5 model instance
    summarizer = AbstractiveSummarizer(model_name="google-t5/t5-small")
    assert isinstance(summarizer._inference_lock, type(threading.RLock()))

    single_results = []
    batch_results = []
    errors = []

    def single_worker():
        try:
            res = summarizer.summarize_text("First article about renewable energy and battery storage systems.")
            single_results.append(res)
        except Exception as e:
            errors.append(f"Single worker error: {e}")

    def batch_worker():
        try:
            res = summarizer.summarize_batch(
                [
                    "Second article on global semiconductor manufacturing facilities and fabrication plants.",
                    "Third article analyzing space exploration initiatives and interstellar telescope designs.",
                ],
                batch_size=2,
            )
            batch_results.append(res)
        except Exception as e:
            errors.append(f"Batch worker error: {e}")

    t1 = threading.Thread(target=single_worker)
    t2 = threading.Thread(target=batch_worker)

    t1.start()
    t2.start()

    t1.join(timeout=20.0)
    t2.join(timeout=20.0)

    assert len(errors) == 0, f"Encountered synchronization errors: {errors}"
    assert len(single_results) == 1
    assert len(single_results[0]["summary"]) > 0
    assert len(batch_results) == 1
    assert len(batch_results[0]) == 2
    assert len(batch_results[0][0]["summary"]) > 0
    assert len(batch_results[0][1]["summary"]) > 0


def test_batch_inference_reentrancy_on_oom_fallback(caplog):
    """Verify that CUDA OOM during batch inference does not deadlock when calling summarize_text."""
    from src.summarizer import AbstractiveSummarizer
    summarizer = AbstractiveSummarizer(model_name="google-t5/t5-small")

    call_count = 0
    real_generate = summarizer.model.generate

    def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate OOM on first attempt (batch generation)
            raise torch.cuda.OutOfMemoryError("CUDA OOM in batch test")
        return real_generate(*args, **kwargs)

    with caplog.at_level(logging.WARNING):
        with patch.object(summarizer.model, "generate", side_effect=mock_generate):
            # Should not deadlock; RLock allows summarize_batch to invoke summarize_text safely
            articles = [
                "Article A discussing environmental policy and forest conservation.",
                "Article B analyzing macroeconomic inflation indicators and monetary policy.",
            ]
            results = summarizer.summarize_batch(articles, batch_size=2)
            assert len(results) == 2
            assert len(results[0]["summary"]) > 0
            assert len(results[1]["summary"]) > 0

    assert call_count == 3
    assert "CUDA OOM during batch generation" in caplog.text
    assert "Batch generation exception" not in caplog.text


def test_legacy_inference_blocks_model_eviction():
    """Verify that a model retrieved via legacy get_model cannot be evicted while inference is running."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        # 1. Retrieve model using legacy direct retrieval (unleased)
        t5_model = mgr.get_model("t5")
        assert not mgr.is_leased("t5")  # No active lease was acquired

        inference_running = threading.Event()
        inference_should_finish = threading.Event()
        inference_completed = threading.Event()

        def legacy_inference_worker():
            with t5_model._inference_lock:
                inference_running.set()
                # Hold the lock simulating in-flight inference
                inference_should_finish.wait(timeout=2.0)
            inference_completed.set()

        t_inf = threading.Thread(target=legacy_inference_worker)
        t_inf.start()
        inference_running.wait(timeout=1.0)

        # 2. While legacy inference is running, another request attempts to load 'bart'.
        # Because max_cached_models=1 and 't5' is executing inference,
        # 't5' must NOT be evicted. Attempting to acquire 'bart' with short timeout must time out.
        with pytest.raises(TimeoutError):
            with mgr.lease_model("bart", timeout=0.1):
                pass

        # Assert t5 was NOT unloaded while inference was running
        assert not t5_model.is_unloaded

        # 3. Allow legacy inference to complete
        inference_should_finish.set()
        t_inf.join(timeout=2.0)
        assert inference_completed.is_set()

        # 4. Now that inference finished and t5 is idle, loading bart succeeds and evicts idle t5
        with mgr.lease_model("bart") as (bart_model, _):
            assert bart_model.model_name == "facebook/bart-large-cnn"
            assert t5_model.is_unloaded


# ==============================================================================
# Priority 4: Deterministic Lifecycle Regression Tests (Zero Arbitrary Sleeps)
# ==============================================================================

def test_eviction_blocked_during_preprocessing():
    """Verify that an inference request paused during preprocessing blocks model eviction."""
    mgr = ModelManager(max_cached_models=1)

    prep_entered = threading.Event()
    prep_can_finish = threading.Event()
    inference_completed = threading.Event()
    inference_errors = []

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        with mgr.lease_model("bart") as (bart_model, _):
            pass

        cached_bart = mgr._cache["bart"]
        orig_summarize = cached_bart.summarize_text

        def mock_summarize_text(text, **kwargs):
            # Preprocessing happens inside _inference_lock
            with cached_bart._inference_lock:
                prep_entered.set()
                prep_can_finish.wait(timeout=2.0)
                return orig_summarize(text, **kwargs)

        cached_bart.summarize_text = mock_summarize_text

        def inference_worker():
            try:
                with mgr.lease_model("bart") as (model, _):
                    model.summarize_text("Sample article text")
            except Exception as e:
                inference_errors.append(str(e))
            finally:
                inference_completed.set()

        t_inf = threading.Thread(target=inference_worker)
        t_inf.start()
        prep_entered.wait(timeout=1.0)

        # While BART is preprocessing under lease and lock, T5 attempt to evict must time out
        with pytest.raises(TimeoutError):
            with mgr.lease_model("t5", timeout=0.1):
                pass

        # Assert BART was not unloaded during preprocessing
        assert not cached_bart.is_unloaded
        assert mgr.is_leased("bart")

        # Allow preprocessing to finish
        prep_can_finish.set()
        t_inf.join(timeout=2.0)
        assert inference_completed.is_set()
        assert len(inference_errors) == 0
        assert not mgr.is_leased("bart")


def test_inference_waiting_on_inference_lock_blocks_eviction():
    """Verify that when a second inference request holds a lease and is waiting on _inference_lock, model cannot be evicted."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        with mgr.lease_model("bart") as (bart_model, _):
            pass

        cached_bart = mgr._cache["bart"]
        lock_held = threading.Event()
        release_lock = threading.Event()
        worker2_waiting = threading.Event()
        worker2_done = threading.Event()

        def lock_holder():
            with cached_bart._inference_lock:
                lock_held.set()
                release_lock.wait(timeout=2.0)

        def second_inference():
            with mgr.lease_model("bart") as (model, _):
                worker2_waiting.set()
                # Blocks waiting for cached_bart._inference_lock
                model.summarize_text("Second request")
            worker2_done.set()

        t1 = threading.Thread(target=lock_holder)
        t1.start()
        lock_held.wait(timeout=1.0)

        t2 = threading.Thread(target=second_inference)
        t2.start()
        worker2_waiting.wait(timeout=1.0)

        # Thread 2 holds a lease on bart while waiting on _inference_lock.
        # Attempting to evict bart to load t5 must time out.
        assert mgr.get_lease_count("bart") >= 1
        with pytest.raises(TimeoutError):
            with mgr.lease_model("t5", timeout=0.1):
                pass

        assert not cached_bart.is_unloaded

        # Release first lock and wait for thread 2 to complete
        release_lock.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        assert worker2_done.is_set()


def test_unload_model_waits_for_active_inference_to_complete():
    """Verify that unload_model() blocks until an active inference request completes."""
    from src.summarizer import AbstractiveSummarizer

    summarizer = AbstractiveSummarizer(model_name="google-t5/t5-small")
    inference_started = threading.Event()
    allow_finish = threading.Event()
    unload_started = threading.Event()
    unload_finished = threading.Event()

    orig_prepare = summarizer._prepare_input_text

    def mock_prepare(text):
        inference_started.set()
        allow_finish.wait(timeout=2.0)
        return orig_prepare(text)

    summarizer._prepare_input_text = mock_prepare

    def inference_worker():
        summarizer.summarize_text("Article for testing unload synchronization.")

    def unload_worker():
        inference_started.wait(timeout=1.0)
        unload_started.set()
        summarizer.unload_model()
        unload_finished.set()

    t_inf = threading.Thread(target=inference_worker)
    t_unload = threading.Thread(target=unload_worker)

    t_inf.start()
    t_unload.start()

    unload_started.wait(timeout=1.0)
    assert not unload_finished.is_set()
    assert summarizer.model is not None
    assert summarizer.tokenizer is not None

    # Now allow inference to complete
    allow_finish.set()
    t_inf.join(timeout=5.0)
    t_unload.join(timeout=5.0)

    assert unload_finished.is_set()
    assert summarizer.model is None
    assert summarizer.tokenizer is None


def test_legacy_retrieval_safe_proxy_prevents_use_after_unload():
    """Verify that a SafeModelProxy acquired via legacy get_model() transparently re-leases after eviction."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        # 1. Retrieve legacy model reference
        t5_proxy = mgr.get_model("t5")
        first_instance = t5_proxy._last_instance
        assert first_instance is not None
        assert not first_instance.is_unloaded

        # 2. Evict T5 by leasing BART
        with mgr.lease_model("bart") as (bart_model, _):
            assert "bart" in mgr._cache
            assert "t5" not in mgr._cache
            assert first_instance.is_unloaded

        # 3. Caller uses the previously retrieved t5_proxy for inference.
        # SafeModelProxy must safely re-lease and reload T5 without throwing use-after-unload error.
        res = t5_proxy.summarize_text("Testing transparent re-leasing after eviction")
        assert "t5" in res["summary"]
        assert not t5_proxy._last_instance.is_unloaded


def test_batch_preprocessing_and_oom_fallback_safety():
    """Verify batch inference holds _inference_lock during preprocessing and sequential fallback."""
    from src.summarizer import AbstractiveSummarizer

    summarizer = AbstractiveSummarizer(model_name="google-t5/t5-small")
    prep_lock_acquired = []

    orig_prepare = summarizer._prepare_input_text

    def mock_prepare(text):
        # Verify that _inference_lock is already held during preprocessing
        acquired = summarizer._inference_lock.acquire(blocking=False)
        if not acquired:
            prep_lock_acquired.append(False)
        else:
            # Reentrancy check: since it's an RLock, it should succeed if the current thread owns it
            summarizer._inference_lock.release()
            prep_lock_acquired.append(True)
        return orig_prepare(text)

    summarizer._prepare_input_text = mock_prepare

    articles = [
        "First article for batch safety verification.",
        "Second article for batch safety verification.",
    ]
    results = summarizer.summarize_batch(articles, batch_size=2)
    assert len(results) == 2
    assert all(prep_lock_acquired)
    assert len(prep_lock_acquired) == 2


def test_no_leaked_leases_or_deadlocks_after_exception():
    """Verify that exceptions during inference do not leak leases or cause deadlocks."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        # 1. Lease model and raise exception inside the lease context
        with pytest.raises(RuntimeError, match="Simulated inference error"):
            with mgr.lease_model("bart") as (model, _):
                assert mgr.is_leased("bart")
                raise RuntimeError("Simulated inference error")

        # 2. Verify lease was cleaned up immediately
        assert not mgr.is_leased("bart")
        assert mgr.get_lease_count("bart") == 0

        # 3. Subsequent lease operations on both models must succeed without deadlock or timeout
        with mgr.lease_model("bart") as (bart_model, _):
            assert bart_model.model_name == "facebook/bart-large-cnn"

        with mgr.lease_model("t5") as (t5_model, _):
            assert t5_model.model_name == "google-t5/t5-small"

        assert not mgr.is_leased("t5")


# ==============================================================================
# SafeModelProxy Resource-Lifecycle & Boundary Security Tests
# ==============================================================================

def test_proxy_raw_model_access_rejected():
    """Verify that accessing proxy.model raises RuntimeError and does not expose raw weights."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        proxy = mgr.get_model("t5")
        with pytest.raises(RuntimeError, match="Direct access to raw model weights on 'SafeModelProxy' is disabled"):
            _ = proxy.model


def test_proxy_raw_tokenizer_access_rejected():
    """Verify that accessing proxy.tokenizer raises RuntimeError and does not expose raw tokenizer."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        proxy = mgr.get_model("t5")
        with pytest.raises(RuntimeError, match="Direct access to raw tokenizer on 'SafeModelProxy' is disabled"):
            _ = proxy.tokenizer


def test_proxy_getattr_bound_methods_and_internals_rejected():
    """Verify that __getattr__ rejects bound methods and raw model internals with clear guidance."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        proxy = mgr.get_model("t5")

        unsupported_attributes = [
            "generate",
            "forward",
            "_load_model_and_tokenizer",
            "to",
            "eval",
            "parameters",
            "state_dict",
        ]
        for attr in unsupported_attributes:
            with pytest.raises(AttributeError, match="does not expose raw model internal or method"):
                _ = getattr(proxy, attr)


def test_proxy_safe_metadata_attributes_accessible():
    """Verify that whitelisted safe metadata attributes can be safely read on SafeModelProxy."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        proxy = mgr.get_model("t5")
        assert proxy.model_name == "google-t5/t5-small"
        assert proxy.device is not None
        assert proxy.is_t5 is True
        assert proxy.max_input_length > 0
        assert isinstance(proxy.generation_params, dict)
        assert proxy.cold_latency_sec is not None or proxy.cold_latency_sec is None
        assert isinstance(proxy.warm_latencies, list)
        assert isinstance(proxy.inference_count, int)
        assert isinstance(proxy.is_unloaded, bool)
        assert isinstance(proxy.is_inferring, bool)


def test_proxy_exceptions_do_not_leak_leases():
    """Verify that an exception raised during proxy.summarize_text does not leak leases or cause deadlock."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        proxy = mgr.get_model("bart")

        def exploding_summarize(*args, **kwargs):
            raise ValueError("Explosion during proxy inference")

        cached_model = mgr._cache["bart"]
        with patch.object(cached_model, "summarize_text", side_effect=exploding_summarize):
            with pytest.raises(ValueError, match="Explosion during proxy inference"):
                proxy.summarize_text("Sample text")

        # Lease must be 0 and is_leased False
        assert not mgr.is_leased("bart")
        assert mgr.get_lease_count("bart") == 0

        # Subsequent inference on another model succeeds without timing out
        t5_proxy = mgr.get_model("t5")
        res = t5_proxy.summarize_text("Testing clean recovery after error")
        assert "t5" in res["summary"]


def test_proxy_supported_legacy_apis_continue_working():
    """Verify that all supported methods on SafeModelProxy work as expected."""
    mgr = ModelManager(max_cached_models=1)

    with patch("app.AbstractiveSummarizer", side_effect=lambda model_name, **kw: MockSummarizer(model_name)):
        proxy = mgr.get_model("t5")

        # 1. summarize_text
        res_single = proxy.summarize_text("Single article test")
        assert len(res_single["summary"]) > 0

        # 2. summarize_batch
        res_batch = proxy.summarize_batch(["Article 1", "Article 2"])
        assert len(res_batch) == 2

        # 3. warmup
        warmup_res = proxy.warmup("Warmup article")
        assert warmup_res >= 0.0

        # 4. unload_model
        assert not proxy.is_unloaded
        proxy.unload_model()
        assert proxy.is_unloaded

