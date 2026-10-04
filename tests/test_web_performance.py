"""Batch equivalence and failure cleanup for the live upload path."""
from dataclasses import replace
import sys
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.web.backend import SubprocessBackend
from visual_peer_inspector.web.core import InspectionRequest, View
from visual_peer_inspector.web.deployment import WorkerSettings
from visual_peer_inspector.web.encoder_worker import encode_batches


def test_encoder_reuses_instance_without_changing_batch_boundaries():
    sizes = []
    class Model:
        def encode(self, crops):
            sizes.append(len(crops))
            vectors = np.asarray(crops, dtype=np.float32).reshape(-1, 1)
            return EncodedBatch(vectors, tuple(v.copy() for v in vectors), 16, 1)
    model = Model()
    result = encode_batches(model, list(range(11)))
    assert sizes == [4, 4, 3]
    np.testing.assert_array_equal(result.global_features.ravel(), np.arange(11))
    np.testing.assert_array_equal(np.asarray(result.patch_features).ravel(), np.arange(11))
    for count in (0, 41):
        with pytest.raises(ValueError):
            encode_batches(model, list(range(count)))


def test_gate_released_when_initial_cache_cleanup_fails(monkeypatch, tmp_path):
    backend = SubprocessBackend(WorkerSettings(sys.executable), worker_module='tests.web_worker_stub')
    from visual_peer_inspector.web.core import load_views
    paths = [tmp_path / f'{i}.png' for i in range(2)]
    for i, path in enumerate(paths):
        Image.new('RGB', (64, 64), i).save(path)
    request = InspectionRequest(load_views(paths))
    cleanup = Mock(side_effect=OSError('test cleanup failure'))
    monkeypatch.setattr(backend, '_cleanup_expired_locked', cleanup)
    with pytest.raises(OSError):
        backend.inspect(request, lambda _: None)
    assert not backend._gate.locked()
    cleanup.side_effect = None
    result = backend.inspect(request, lambda _: None)
    assert result.provenance['timing']['request_seconds'] > 0
    assert not backend._gate.locked()


def test_invalid_token_releases_gate_without_removing_owned_session():
    backend = SubprocessBackend(WorkerSettings(sys.executable), worker_module='tests.web_worker_stub')
    request = InspectionRequest(())
    backend._sessions[request.session_id] = 'owned'
    with pytest.raises(ValueError, match='cached session'):
        backend.inspect(replace(request, cache_token='wrong'), lambda _: None)
    assert not backend._gate.locked()
    assert backend._sessions[request.session_id] == 'owned'


def test_page_initialisation_does_not_queue_behind_inference():
    from visual_peer_inspector.web.ui import build_app
    app = build_app()
    load = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == 'initialise')
    assert load.queue is False
    assert load.show_progress == 'hidden'
    inspect = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == 'do_inspect')
    assert inspect.concurrency_id == 'inspection'
    assert inspect.show_progress_on is not None


def test_browser_disconnect_cancels_only_its_own_inspection(tmp_path):
    import gradio as gr
    from visual_peer_inspector.web.core import Session, load_views
    from visual_peer_inspector.web.ui import build_app
    backend = Mock()
    app = build_app(backend)
    callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    browser = gr.Request(session_hash='browser-a')
    paths = [tmp_path / f'{i}.png' for i in range(2)]
    for i, path in enumerate(paths):
        Image.new('RGB', (64, 64), i).save(path)
    def inspect(request, progress):
        callbacks['browser_disconnected'](gr.Request(session_hash='browser-b'))
        backend.cancel.assert_not_called()
        callbacks['browser_disconnected'](browser)
        backend.cancel.assert_called_once_with(request.request_id)
        progress('Finding objects')
        pytest.fail('Disconnected inspection must not continue')
    backend.inspect.side_effect = inspect
    state = Session(load_views(paths))
    result = callbacks['do_inspect'](state, browser_request=browser)
    assert state.active_request_id is None
    assert state.result is None
    assert any('browser session closed' in str(v) for v in result.values())
    callbacks['browser_disconnected'](browser)
    assert backend.cancel.call_count == 1
