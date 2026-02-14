from multiobjective.ga.individual import Individual


def test_individual_genome_size(cfg):
    ind = Individual(cfg)

    assert len(ind.genome) == cfg.genome_cfg.size
    assert all(g in (0, 1) for g in ind.genome)


def test_genome_is_binary(cfg):
    ind = Individual(cfg)

    assert set(ind.genome).issubset({0, 1})


def test_decode_genome_returns_dict(cfg):
    ind = Individual(cfg)

    ind.genome = [1] * cfg.genome_cfg.size
    hyperparams = ind._decode_genome()

    assert isinstance(hyperparams, dict)
    assert len(hyperparams) > 0

def test_model_can_be_built_from_genome(cfg):
    ind = Individual(cfg)

    ind.genome = [1] * cfg.genome_cfg.size
    model = ind._build_model()

    assert model is not None


def test_same_genome_same_hyperparams(cfg):
    genome = [1, 0] * (cfg.genome_cfg.size // 2)

    ind1 = Individual(cfg, genome=genome.copy())
    ind2 = Individual(cfg, genome=genome.copy())

    h1 = ind1._decode_genome()
    h2 = ind2._decode_genome()

    assert h1 == h2


def test_mutation_all_bits_flip(cfg):
    ind = Individual(cfg)
    ind.genome = [0] * cfg.genome_cfg.size

    ind.mutation(rate=1.0)

    assert all(g == 1 for g in ind.genome)


def test_mutation_changes_genome(cfg):
    ind = Individual(cfg)
    original = ind.genome.copy()

    ind.mutation(rate=1.0)

    assert ind.genome != original


def test_crossover_produces_valid_children(cfg):
    p1 = Individual(cfg)
    p2 = Individual(cfg)

    c1, c2 = p1.crossover(p2)

    assert len(c1.genome) == cfg.genome_cfg.size
    assert len(c2.genome) == cfg.genome_cfg.size
    assert set(c1.genome).issubset({0, 1})
    assert set(c2.genome).issubset({0, 1})


def test_individual_evaluate_sets_metrics(cfg, fake_data):
    ind = Individual(cfg)

    ind.evaluate(*fake_data, epochs=1, batch_size=4)

    expected_metrics = {
        "loss",
        "accuracy",
        "f1_score",
        "auc",
        "weights_norm",
        "latency",
    }

    assert expected_metrics.issubset(ind.metrics.keys())


def test_evaluate_builds_model_and_hyperparams(cfg, fake_data):
    ind = Individual(cfg)

    ind.evaluate(*fake_data, epochs=1, batch_size=4)

    assert ind.model is not None
    assert ind.hyperparams is not None


def test_full_pipeline_with_small_data(cfg, fake_data):
    ind = Individual(cfg)

    ind.evaluate(*fake_data, epochs=1, batch_size=4)


    assert ind.metrics["accuracy"][1] >= 0.0
