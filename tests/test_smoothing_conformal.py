import numpy as np

from wikicat.conformal import ConformalCalibrator, nonconformity, set_report, topk_hit_rate
from wikicat.hierarchy import CategoryGraph
from wikicat.smoothing import neighbour_matrix, smooth_prototypes


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_neighbours_are_parents_and_siblings():
    names = ["Birds", "Sparrows", "Crows", "Cats"]
    graph = CategoryGraph({"Sparrows": ["Birds"], "Crows": ["Birds"], "Cats": ["Animals"]})
    a = neighbour_matrix(names, graph).toarray()
    # Sparrows: parent Birds (half) and sibling Crows (half); Cats has no neighbour in the vocabulary.
    assert np.allclose(a[1], [0.5, 0, 0.5, 0]) and not a[3].any() and not a[0].any()


def test_tail_category_borrows_from_its_family():
    rng = np.random.default_rng(0)
    names = ["Birds", "Sparrows", "Crows"]
    graph = CategoryGraph({"Sparrows": ["Birds"], "Crows": ["Birds"]})
    text = unit(rng.normal(size=(3, 16)))
    bird_photo = unit(rng.normal(size=16))
    embs = unit(bird_photo + 0.05 * rng.normal(size=(20, 16)))
    labels = [[0, 2]] * 20  # many photos of Birds and Crows, none of Sparrows

    plain = smooth_prototypes(text, names, graph, embs, labels, gamma=0.0, use_images=False)
    assert np.allclose(plain, text, atol=1e-6)
    smoothed = smooth_prototypes(text, names, graph, embs, labels, kappa=5.0, gamma=0.5)
    assert np.allclose(np.linalg.norm(smoothed, axis=1), 1, atol=1e-5)
    # Sparrows has no photos but moves towards what its parent and sibling look like.
    family = unit(smoothed[0] + smoothed[2])
    assert smoothed[1] @ family > text[1] @ family + 0.1
    # Crows has photos, so it leans on its own centroid more than on its name.
    assert smoothed[2] @ text[2] < 0.9


def test_conformal_threshold_and_coverage():
    names = ["a", "b", "c", "d"]
    assert nonconformity(np.array([0.9, 0.2]), ["a", "b"], {"b"}) == 0.8
    assert nonconformity(np.array([0.9]), ["a"], {"z"}) == 1.0
    cal = ConformalCalibrator({"with_metadata": [0.1, 0.2, 0.3, 0.4], "image_only": [0.5] * 9 + [1.0]})
    # ceil(5 * 0.8) = 4th smallest score 0.4 -> threshold 0.6
    assert abs(cal.threshold(0.2, "with_metadata") - 0.6) < 1e-9
    # Too few calibration images for alpha 0.05: the set must include every candidate.
    assert cal.threshold(0.05, "image_only") == 0.0
    assert list(cal.predict_set(np.array([0.7, 0.65, 0.1]), True, 0.2)) == [0, 1]

    # Exchangeable synthetic data: empirical coverage reaches the target.
    rng = np.random.default_rng(1)

    def sample(n):
        scored, gold = [], []
        for _ in range(n):
            final = rng.uniform(size=4)
            scored.append({"cands": np.arange(4), "final": final})
            gold.append({names[int(rng.integers(4))]} if rng.uniform() < 0.9 else {"zz"})
        return scored, gold

    cal_scored, cal_gold = sample(500)
    cal = ConformalCalibrator({"with_metadata": [nonconformity(r["final"], names, g)
                                                 for r, g in zip(cal_scored, cal_gold)]})
    test_scored, test_gold = sample(2000)
    rep = set_report(cal, test_scored, [True] * 2000, test_gold, names, alphas=(0.2, 0.4))
    for v in rep.values():
        assert v["coverage"] >= v["target"] - 0.03
    assert rep["0.4"]["mean_size"] < rep["0.2"]["mean_size"]
    assert 0 < topk_hit_rate(test_scored, test_gold, names, 1) < topk_hit_rate(test_scored, test_gold, names, 3)


def test_choose_alpha_prefers_strongest_reviewable_guarantee():
    cal = ConformalCalibrator({"with_metadata": [0.5]})
    assert cal.choose_alpha({0.1: 40, 0.2: 12, 0.3: 6, 0.4: 3}) == 0.3
    assert cal.choose_alpha({0.1: 40, 0.2: 30}) == 0.2
