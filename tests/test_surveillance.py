import numpy as np

from surveillance_sim import (
    hospital_catchments,
    simulate_epidemic,
    simulate_gp_surveillance,
    simulate_hospital_surveillance,
)


def _toy_outbreak():
    out = np.zeros((30, 3), dtype=int)
    out[5:, 0] = 100
    out[10:, 1] = 50
    return out


def test_hospital_no_infections_returns_none():
    out = np.zeros((30, 3), dtype=int)
    assert simulate_hospital_surveillance(out, [[0], [1], [0, 1]], 2, 1.0, 1.0, 1.0, rng=0) is None


def test_hospital_certain_detection_on_first_infection_day():
    res = simulate_hospital_surveillance(_toy_outbreak(), [[0], [1], [0, 1]], 2, 1.0, 1.0, 1.0, rng=0)
    assert res.detection_day == 5
    assert res.cumulative_infections == 100
    assert res.n_locations_infected == 1
    assert res[1] == 5  # list-style indexing still works


def test_hospital_no_participating_hospitals_never_detects():
    assert simulate_hospital_surveillance(_toy_outbreak(), [[0], [1], [0, 1]], 2, 0.0, 1.0, 1.0, rng=0) is None


def test_gp_certain_detection_on_first_infection_day():
    out = np.zeros((30, 1), dtype=int)
    out[5:, 0] = 100
    res = simulate_gp_surveillance(out, np.array([1000]), np.zeros(52), n_gps=5, p_symp=1.0,
                                   healthcare_seeking_p=1.0, init_week=0,
                                   n_weekly_samples=7000, rng=0)
    assert res.detection_day == 5


def test_gp_no_infections_returns_none():
    out = np.zeros((30, 2), dtype=int)
    res = simulate_gp_surveillance(out, np.array([1000, 2000]), np.full(52, 1e-3), n_gps=5,
                                   p_symp=0.6, healthcare_seeking_p=0.2, init_week=0,
                                   n_weekly_samples=10, rng=0)
    assert res is None


def test_hospital_catchments_fallback_to_nearest():
    locs = np.array([[0, 0], [100_000, 0]])
    hosps = np.array([[1_000, 0], [2_000, 0], [50_000, 0]])
    assert hospital_catchments(locs, hosps, radius=10_000) == [[0, 1], [2]]


def test_simulate_epidemic_shape_and_threshold():
    mobility = np.array([[0.4, 0.1], [0.1, 0.4]])
    out = simulate_epidemic(0, alpha=0.5, beta=0.25, mobility=mobility, n_days=100,
                            population=np.array([10_000, 10_000]), extinction_threshold=100,
                            poisson=True, rng=1)
    assert out.shape == (100, 2)
    assert out.sum() >= 100
