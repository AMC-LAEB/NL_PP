import numpy as np
import pytest

from surveillance_sim.fitting import MobilityLikelihood, fit_mobility
from surveillance_sim.netherlands import load_netherlands


@pytest.fixture(scope="module")
def nl():
    return load_netherlands()


@pytest.fixture(scope="module")
def fit_inputs(nl):
    return dict(
        distance_km=nl.distance_km(),
        population=nl.population,
        town_region=nl.towns["municipality"],
        work_location=nl.commuting_work_location,
        home_location=nl.commuting_home_location,
    )


def test_data_shapes(nl):
    assert len(nl.towns) == len(nl.population) == nl.distance_km().shape[0]
    assert len(nl.close_hospitals()) == len(nl.towns)
    assert all(len(h) >= 1 for h in nl.close_hospitals())
    ili = nl.background_ili()
    assert ili.shape == (52,)
    assert np.all((ili > 0) & (ili < 0.01))


def test_likelihood_is_finite(fit_inputs):
    lik = MobilityLikelihood(**fit_inputs)
    assert np.isfinite(lik.log_likelihood(np.array([1.5, 0.6, 5, -0.5, -2])))


def test_fit_runs(fit_inputs):
    fit = fit_mobility(**fit_inputs, n_walkers=10, n_steps=3, burnin=1, rng=0, progress=False)
    assert fit.chain.shape == (3, 10, 5)
    assert fit.samples.shape == (20, 5)
    assert set(fit.median()) == {"dist_exp", "pop_exp", "offset", "epsilon", "log_sigma"}
