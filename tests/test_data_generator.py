from src.data_generator import GeneratorConfig, generate_jobs, validate_jobs


def test_generator_is_valid_and_reproducible():
    a = generate_jobs(GeneratorConfig(seed=123))
    b = generate_jobs(GeneratorConfig(seed=123))
    validate_jobs(a)
    assert not a.empty
    assert a.equals(b)
    assert a["arrival_time"].is_monotonic_increasing
    assert (a["processing_time"] > 0).all()
    assert (a["due_time"] >= a["arrival_time"]).all()
