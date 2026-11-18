import numpy as np
import cv2
import pytest

from vad.data import common, flow, prepare
from vad.data.clips import VideoStore


# ---------------------------------------------------------------- pure logic
def test_segments_from_labels():
    assert common.segments_from_labels([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert common.segments_from_labels([0, 0, 0]) == []
    assert common.segments_from_labels([1, 1, 1]) == [(0, 2)]
    assert common.segments_from_labels([]) == []


def test_split_train_val():
    ids = [f"v{i:02d}" for i in range(20)]
    tr, va = common.split_train_val(ids, 0.1)
    assert len(va) == 2 and set(tr) | set(va) == set(ids) and not set(tr) & set(va)
    assert common.split_train_val(ids, 0.1) == (tr, va)              # deterministic
    assert len(common.split_train_val([f"v{i}" for i in range(16)], 0.1)[1]) == 2
    assert common.split_train_val(["a"], 0.1) == (["a"], [])
    tr, va = common.split_train_val(["a", "b"], 0.5)
    assert len(tr) == 1 and len(va) == 1


def test_clip_indices_causal():
    idx = common.clip_frame_indices(3, 8, 50)
    assert idx.tolist() == [0, 0, 0, 0, 0, 0, 1, 2]
    for t in (0, 1, 5, 30, 49):
        assert common.clip_frame_indices(t, 8, 50).max() <= max(t - 1, 0)
    assert common.train_targets(20, 8).tolist() == list(range(8, 20))


def test_flow_quantize_roundtrip():
    f = np.random.uniform(-flow.FLOW_MAX, flow.FLOW_MAX, (2, 16, 16)).astype(np.float32)
    err = np.abs(flow.dequantize(flow.quantize(f)) - f).max()
    assert err <= flow.FLOW_MAX / 127 / 2 + 1e-4
    assert flow.quantize(np.zeros((2, 4, 4), np.float32)).min() == 128


# ---------------------------------------------------------------- synthetic dataset
def _const_frames(n, step=10):
    return [np.full((256, 256, 3), step * i, np.uint8) for i in range(n)]


@pytest.fixture()
def toy(tmp_path):
    root = tmp_path / "processed"
    prepare.write_video(root, "toy", "train", "a", _const_frames(20), 25.0)
    prepare.write_video(root, "toy", "train", "b", _const_frames(20), 25.0)
    labels = np.zeros(30, np.uint8); labels[10:15] = 1; labels[20:25] = 1
    masks = np.zeros((30, 64, 64), np.uint8); masks[10:15, 10:20, 10:20] = 1; masks[20:25, 5:9, 5:9] = 1
    prepare.write_video(root, "toy", "test", "t", _const_frames(30, 5), 25.0, labels=labels, masks=masks)
    return root


def test_store_clip_is_causal_and_shaped(toy):
    s = VideoStore(toy, "toy", "train", "a", clip_len=4, img_size=224)
    d = s.get_clip(10)
    assert d["clip"].shape == (4, 3, 224, 224) and d["target"].shape == (3, 224, 224)
    means = d["clip"].mean(axis=(1, 2, 3)) * 255
    assert np.allclose(means, [60, 70, 80, 90], atol=3)               # frames 6..9, never frame 10
    assert abs(d["target"].mean() * 255 - 100) < 3


def test_eval_start_padding_and_gt(toy):
    s = VideoStore(toy, "toy", "test", "t", clip_len=4, img_size=128)
    d = s.get_clip(0)                                                 # replicate-padded start
    assert np.allclose(d["clip"].mean(axis=(1, 2, 3)) * 255, 0, atol=2)
    lab, m = s.labels(), s.masks()
    assert lab.shape == (30,) and common.segments_from_labels(lab) == [(10, 14), (20, 24)]
    assert m.shape == (30, 128, 128) and m[12].any() and not m[0].any()
    assert s.meta.fps == 25.0 and s.meta.has_masks


def test_list_videos(toy):
    assert common.list_videos(toy, "toy", "train") == ["a", "b"]
    assert common.list_videos(toy, "toy", "nope") == []


def test_flow_online_equals_precomputed(tmp_path):
    rng = np.random.RandomState(0)
    base = cv2.GaussianBlur(rng.randint(0, 255, (300, 400), np.uint8), (0, 0), 3)
    frames = [cv2.cvtColor(base[20:276, 4 * i:4 * i + 256], cv2.COLOR_GRAY2BGR) for i in range(12)]
    root, froot = tmp_path / "p", tmp_path / "f"
    prepare.write_video(root, "toy", "train", "m", frames, 25.0)
    n = flow.precompute_video(common.video_dir(root, "toy", "train", "m"),
                              common.flow_path(froot, "toy", "train", "m"))
    assert n == 11
    a = VideoStore(root, "toy", "train", "m", 6, 224, "online", froot).get_clip(8)["flow"]
    b = VideoStore(root, "toy", "train", "m", 6, 224, "precomputed", froot).get_clip(8)["flow"]
    assert a.shape == (5, 2, 128, 128) and np.array_equal(a, b)
    assert np.abs(a[..., 0, :, :]).mean() > 0.5                        # real horizontal motion detected


# ---------------------------------------------------------------- converters on fake raw data
def _vid(path, n, size=(160, 120), fourcc="MJPG", fps=25):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8))
    w.release()


def test_prepare_ped2(tmp_path):
    raw = tmp_path / "raw" / "UCSDped2"
    for sp, vids in (("Train", ["Train001"]), ("Test", ["Test001"])):
        for v in vids:
            (raw / sp / v).mkdir(parents=True)
            for i in range(1, 11):
                cv2.imwrite(str(raw / sp / v / f"{i:03d}.tif"), np.full((240, 360), i, np.uint8))
    gt = raw / "Test" / "Test001_gt"; gt.mkdir()
    for i in range(1, 11):
        m = np.zeros((240, 360), np.uint8)
        if 4 <= i <= 7:
            m[50:100, 50:100] = 255
        cv2.imwrite(str(gt / f"{i:03d}.bmp"), m)
    out = tmp_path / "out"
    prepare.prepare_ped2(tmp_path / "raw", out)
    s = VideoStore(out, "ped2", "test", "Test001", 4, 224)
    assert s.num_frames == 10 and common.segments_from_labels(s.labels()) == [(3, 6)]
    assert s.masks()[4].sum() > 0 and s.meta.fps == 10.0
    assert common.list_videos(out, "ped2", "train") == ["Train001"]


def test_prepare_avenue(tmp_path):
    from scipy.io import savemat
    raw = tmp_path / "raw" / "Avenue Dataset"
    (raw / "training_videos").mkdir(parents=True); (raw / "testing_videos").mkdir()
    _vid(raw / "training_videos" / "01.avi", 12); _vid(raw / "testing_videos" / "01.avi", 12)
    vol = np.empty((1, 12), dtype=object)
    for i in range(12):
        m = np.zeros((120, 160), np.uint8)
        if 5 <= i <= 8:
            m[10:40, 10:40] = 1
        vol[0, i] = m
    (raw / "gt").mkdir(); savemat(str(raw / "gt" / "1_label.mat"), {"volLabel": vol})
    out = tmp_path / "out"
    prepare.prepare_avenue(tmp_path / "raw", out)
    s = VideoStore(out, "avenue", "test", "01", 4, 224)
    assert s.num_frames == 12 and common.segments_from_labels(s.labels()) == [(5, 8)]
    assert s.meta.fps == 25.0 and s.masks().shape == (12, 224, 224)


def test_prepare_shanghaitech(tmp_path):
    raw = tmp_path / "raw" / "shanghaitech"
    (raw / "training" / "frames" / "01_001").mkdir(parents=True)
    (raw / "testing" / "frames" / "01_0014").mkdir(parents=True)
    (raw / "testing" / "test_frame_mask").mkdir(); (raw / "testing" / "test_pixel_mask").mkdir()
    for i in range(10):
        cv2.imwrite(str(raw / "training/frames/01_001" / f"{i:03d}.jpg"), np.full((120, 160, 3), i, np.uint8))
        cv2.imwrite(str(raw / "testing/frames/01_0014" / f"{i:03d}.jpg"), np.full((120, 160, 3), i, np.uint8))
    lab = np.zeros(10, np.uint8); lab[6:9] = 1
    np.save(raw / "testing/test_frame_mask/01_0014.npy", lab)
    pm = np.zeros((10, 120, 160), np.uint8); pm[6:9, 20:40, 20:40] = 1
    np.save(raw / "testing/test_pixel_mask/01_0014.npy", pm)
    out = tmp_path / "out"
    prepare.prepare_shanghaitech(tmp_path / "raw", out)
    s = VideoStore(out, "shanghaitech", "test", "01_0014", 4, 224)
    assert common.segments_from_labels(s.labels()) == [(6, 8)] and s.masks()[7].any()
    assert common.list_videos(out, "shanghaitech", "train") == ["01_001"]


def test_prepare_ucf_crime(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "Anomaly").mkdir(); (raw / "Testing_Normal_Videos_Anomaly").mkdir()
    _vid(raw / "Anomaly" / "Abuse028_x264.mp4", 20, fourcc="mp4v", fps=30)
    _vid(raw / "Testing_Normal_Videos_Anomaly" / "Normal_Videos_015_x264.mp4", 20, fourcc="mp4v", fps=30)
    (raw / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text(
        "Abuse028_x264.mp4  Abuse  4  11  -1  -1\n")
    out = tmp_path / "out"
    prepare.prepare_ucf_crime(raw, out, stride=2)
    assert common.list_videos(out, "ucf_crime", "test") == ["Abuse028_x264", "Normal_Videos_015_x264"]
    s = VideoStore(out, "ucf_crime", "test", "Abuse028_x264", 4, 224)
    assert s.meta.stride == 2 and s.meta.fps == pytest.approx(15.0) and s.num_frames == 10
    assert s.labels().sum() > 0 and s.masks() is None
    assert VideoStore(out, "ucf_crime", "test", "Normal_Videos_015_x264", 4, 224).labels().sum() == 0


def test_pack_unpack(toy, tmp_path):
    prepare.pack(toy, tmp_path / "arch", "toy")
    prepare.unpack(tmp_path / "restored", tmp_path / "arch", "toy")
    assert common.list_videos(tmp_path / "restored", "toy", "train") == ["a", "b"]
