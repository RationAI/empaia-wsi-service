import time

import psutil
import requests

slide_ids = [
    "50f3010ed9a55f04b2e0d88cd19c6923",  # Aperio
    "d3fc669ff08d57a4a409340d54d6bf4f",  # Mirax
]


def get_memory_used_in_mb():
    return dict(psutil.virtual_memory()._asdict())["used"] / 1e6


def test_thumbnail_cache_no_additional_memory_usage_after_first_thumbnail_request():
    for slide_id in slide_ids:
        r = requests.get(f"http://localhost:8080/v1/slides/{slide_id}/thumbnail/max_size/500/500")
        assert r.status_code == 200
    memory_usage_after_first_thumbnail_request = get_memory_used_in_mb()
    for _ in range(5):
        for slide_id in slide_ids:
            r = requests.get(f"http://localhost:8080/v1/slides/{slide_id}/thumbnail/max_size/500/500")
            assert r.status_code == 200
    memory_usage_after_addtional_thumbnail_requests = get_memory_used_in_mb()
    assert (memory_usage_after_addtional_thumbnail_requests - memory_usage_after_first_thumbnail_request) < 20.0


def test_thumbnail_cache_speedup_test():
    time.sleep(6)  # make sure slide is closed
    start = time.time()
    r = requests.get("http://localhost:8080/v1/slides/50f3010ed9a55f04b2e0d88cd19c6923/thumbnail/max_size/500/500")
    time_first = time.time() - start
    assert r.status_code == 200
    start = time.time()
    r = requests.get("http://localhost:8080/v1/slides/50f3010ed9a55f04b2e0d88cd19c6923/thumbnail/max_size/500/500")
    time_second = time.time() - start
    assert r.status_code == 200
    speedup = time_first / time_second
    assert speedup > 4.0  # check speedup at least 4x, should usually be more like 10x