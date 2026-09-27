import numpy as np
import pytest

from order_analytics.sources.olist import _haversine_km


def test_haversine_sao_paulo_to_rio_is_about_360_km():
    km = _haversine_km(np.array([-23.55]), np.array([-46.63]), np.array([-22.91]), np.array([-43.17]))
    assert km[0] == pytest.approx(360, abs=10)


def test_haversine_same_point_is_zero():
    assert _haversine_km(np.array([-10.0]), np.array([-50.0]), np.array([-10.0]), np.array([-50.0]))[0] == 0
