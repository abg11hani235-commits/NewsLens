"""Deterministic Concurrency and Lifecycle Tests for ModelManager.

Verifies:
1. Active lease protection: Models actively leased cannot be evicted.
2. Simultaneous requests switching between BART and T5.
3. Condition variable unblocking upon lease release.
4. Concurrent leases on the same model with accurate reference counting.
5. Timeout safety when all cached models are actively leased.
"""

import time
import threading
from unittest.mock import MagicMock, patch
import pytest

from app import ModelManager


class MockSummarizer:
    """Mock summarizer for deterministic concurrency testing."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.init_time_sec = 0.05
        self.cold_latency_sec = 0.1
        self.is_unloaded = False
        self._inference_lock = threading.RLock()

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


def test_batch_inference_reentrancy_on_oom_fallback():
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

