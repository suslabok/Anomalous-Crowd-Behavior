import json

import numpy as np
import pytest

from vad.eval import metrics as M
from vad.eval import scoring as S
from vad.eval.report import clean, summarize


def test_causal_smooth_values_and_no_lookahead():
    assert np.allclose(S.causal_smooth([0, 3, 6, 9], 3), [0, 1.5, 3, 6])
    x = np.random.RandomState(0).rand(50)
    y = x.copy(); y[30:] += 100                                    # change only the FUTURE
    assert np.allclose(S.causal_smooth(x, 5)[:30], S.causal_smooth(y, 5)[:30])
    assert np.allclose(S.causal_smooth(x, 1), x)


def test_patch_pool_and_topk():
    e = np.zeros((2, 4, 4)); e[0, :2, :2] = 1.0
    p = S.patch_pool(e, 2)
    assert p.shape == (2, 2, 2) and p[0, 0, 0] == 1 and p[1].sum() == 0
    assert np.allclose(S.topk_mean(np.array([[[1., 2.], [3., 4.]]]), 2), [3.5])
    assert np.allclose(S.topk_mean(np.array([[[1., 2.], [3., 4.]]]), 99), [2.5])


def test_normalizer_roundtrip(tmp_path):
    n = S.fit_normalizer([np.array([1., 2., 3.]), np.array([4., 5.])])
    assert n.mean == pytest.approx(3.0) and n.std == pytest.approx(np.sqrt(2.0))
    n.save(tmp_path / "n.json")
    assert S.Normalizer.load(tmp_path / "n.json") == n
    assert S.Normalizer(0.0, 0.0)(np.array([1.0])).shape == (1,)   # zero std does not divide by zero


def test_alarm_min_consecutive():
    s = np.array([0, 5, 0, 5, 5, 5, 0])
    assert S.alarm_signal(s, 1).tolist() == [0, 1, 0, 1, 1, 1, 0]
    assert S.alarm_signal(s, 1, 2).tolist() == [0, 0, 0, 0, 1, 1, 0]
    assert S.choose_threshold([np.arange(101)], 99) == pytest.approx(99.0)


def _alarm(n, on):
    a = np.zeros(n, bool); a[list(on)] = True; return a


def test_time_to_detection_basic():
    lab = np.zeros(40, int); lab[10:20] = 1; lab[30:35] = 1
    a = _alarm(40, [13, 14])                                      # detects segment 1 at +3, misses segment 2
    r = M.time_to_detection(a, lab, fps=10.0)
    assert r[0]["detected"] and r[0]["delay_frames"] == 3 and r[0]["delay_s"] == pytest.approx(0.3)
    assert not r[1]["detected"] and r[1]["delay_frames"] is None


def test_early_alarm_is_false_alarm_not_negative_delay():
    lab = np.zeros(40, int); lab[10:20] = 1
    a = _alarm(40, [5, 15])
    r = M.time_to_detection(a, lab, 10.0)[0]
    assert r["delay_frames"] == 5 and not r["carryover"]          # first alarm AT/AFTER onset
    fa, hours = M.false_alarms(a, lab, 10.0)
    assert fa == 1 and hours == pytest.approx(30 / 10 / 3600)     # the frame-5 alarm; 30 normal frames


def test_carryover_flagged_and_counted_as_false_alarm():
    lab = np.zeros(40, int); lab[10:20] = 1
    a = _alarm(40, range(8, 14))                                  # alarm starts before onset and persists
    r = M.time_to_detection(a, lab, 10.0)[0]
    assert r["detected"] and r["delay_frames"] == 0 and r["carryover"]
    assert M.false_alarms(a, lab, 10.0)[0] == 1


def test_tolerance_allows_signed_delay_and_protects_pre_onset():
    lab = np.zeros(40, int); lab[10:20] = 1
    a = _alarm(40, [8])
    assert not M.time_to_detection(a, lab, 10.0, 0)[0]["detected"]
    r = M.time_to_detection(a, lab, 10.0, 3)[0]
    assert r["detected"] and r["delay_frames"] == -2
    assert M.false_alarms(a, lab, 10.0, 0)[0] == 1 and M.false_alarms(a, lab, 10.0, 3)[0] == 0


def test_alarm_events_are_rising_edges():
    lab = np.zeros(30, int)
    a = _alarm(30, [3, 4, 5, 10, 11])                             # two events
    assert M.false_alarms(a, lab, 10.0)[0] == 2


def test_frame_auc_modes():
    assert M.frame_auc([np.array([0., 1.])], [np.array([0, 1])]) == 1.0
    assert np.isnan(M.frame_auc([np.array([0., 1.])], [np.array([0, 0])]))
    s = [np.array([0., 1., 0., 1.]), np.array([5., 6.])]          # second video single-class
    l = [np.array([0, 1, 0, 1]), np.array([0, 0])]
    assert M.frame_auc(s, l, "macro") == 1.0


def test_pixel_acc_and_blur():
    acc = M.PixelAcc(per_video=1000, seed=1)
    masks = np.zeros((4, 32, 32), np.uint8); masks[:, 8:16, 8:16] = 1
    acc.add(masks.astype(np.float32) + 0.01, masks)
    assert acc.auc() == pytest.approx(1.0)
    assert M.blur_maps(masks.astype(np.float32), 2.0).shape == masks.shape


def test_summarize_end_to_end_recovers_injected_delay():
    rng = np.random.RandomState(0)
    vids = []
    for i in range(3):
        raw = 1.0 + 0.05 * rng.randn(120)
        raw[60 + 4:100] += 3.0                                    # anomaly starts at 60 but is visible from 64
        lab = np.zeros(120, int); lab[60:100] = 1
        vids.append(dict(id=f"v{i}", raw=raw, labels=lab, fps=10.0))
    norm = S.Normalizer(1.0, 0.05)
    m = summarize(vids, norm, thr=5.0, smooth=1, min_consecutive=1, tolerances=[0, 3])
    d = m["detection"]["tol_0"]
    assert d["detection_rate"] == 1.0 and d["delay_frames_mean"] == 4.0 and d["delay_s_mean"] == pytest.approx(0.4)
    # 4 of 40 labelled-anomalous frames still look normal (60-63), so AUC is capped near 0.95
    assert d["n_false_alarms"] == 0 and 0.88 < m["frame_auc_micro_causal"] < 0.99
    assert m["detection"]["tol_3"]["delay_frames_mean"] == 4.0
    json.dumps(clean(m))                                          # must be JSON-serialisable (NaN -> null)


def test_plot_writes_png(tmp_path):
    from vad.eval.plots import plot_score_curves
    lab = np.zeros(50, int); lab[20:30] = 1
    v = dict(id="a", score=np.r_[np.zeros(22), 6 * np.ones(28)], labels=lab, alarm=_alarm(50, range(22, 50)))
    plot_score_curves([v, dict(v, id="b")], 5.0, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 1000


# ---------------------------------------------------------------- sweep
def _synthetic_scores(tmp_path):
    from vad.eval.sweep import save_scores
    rng = np.random.RandomState(1)
    val_raws = [1.0 + 0.1 * rng.randn(80) for _ in range(2)]
    videos = []
    for i in range(3):
        raw = 1.0 + 0.1 * rng.randn(120)
        raw[60 + 3:100] += 4.0
        lab = np.zeros(120, int); lab[60:100] = 1
        videos.append(dict(id=f"v{i}", raw=raw, labels=lab, fps=10.0))
    p = tmp_path / "scores.npz"
    save_scores(p, 8, ["a", "b"], val_raws, videos)
    return p


def test_scores_roundtrip_and_sweep(tmp_path):
    from vad.eval.sweep import load_scores, pick_operating_points, run_sweep
    p = _synthetic_scores(tmp_path)
    val_raws, clip_len, videos = load_scores(p)
    assert clip_len == 8 and len(val_raws) == 2 and [v["id"] for v in videos] == ["v0", "v1", "v2"]
    assert videos[0]["labels"].sum() == 40 and videos[0]["fps"] == 10.0
    rows = run_sweep(val_raws, clip_len, videos, smooths=(1, 5), min_consecs=(1, 3), percentiles=(99, 100), zs=(4, 60))
    assert len(rows) == 2 * 2 * 4
    easy = [r for r in rows if r["smooth"] == 1 and r["min_consecutive"] == 1 and r["rule"] == "z4"][0]
    assert easy["detection_rate"] == 1.0 and easy["n_false_alarms"] == 0
    harsh = [r for r in rows if r["smooth"] == 1 and r["min_consecutive"] == 1 and r["rule"] == "z60"][0]
    assert harsh["detection_rate"] < 1.0                               # z60 is above the injected anomaly (~z40) -> missed
    picks = pick_operating_points(rows, budgets=(0, 2))
    assert picks[0]["n_false_alarms"] <= 0 and picks[0]["detection_rate"] == 1.0
    json.dumps(clean(rows))


def test_tradeoff_plot(tmp_path):
    from vad.eval.plots import plot_tradeoff
    from vad.eval.sweep import load_scores, run_sweep
    val_raws, clip_len, videos = load_scores(_synthetic_scores(tmp_path))
    rows = run_sweep(val_raws, clip_len, videos, smooths=(1,), min_consecs=(1,), percentiles=(99,), zs=(4, 8))
    plot_tradeoff(rows, tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000
