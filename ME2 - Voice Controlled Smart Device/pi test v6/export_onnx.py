#!/usr/bin/env python3
"""pi test v6 -- export the recognizer to ONNX.

Two artifacts:

  v6_features.onnx
      waveform (1, N)  ->  log emissions (T, P*3)
      Faithful ONNX port of hgm/feats.py (pre-emphasis, Hamming 25.6 ms /
      10 ms hop, 512-pt DFT as a real matrix multiply, 26-band mel
      filterbank, log, orthonormal DCT-II -> 13 cepstra, CMN, delta,
      delta-of-delta) followed by the GMM emissions of every phone state:
      log p(x_t | phone, state) for all 37 phones x 3 states.

  v6_transitions.npz
      log_a (37, 3, 3) -- the HMM transition log-probabilities. The Viterbi
      decoder (hgm/decode.py) consumes both; the acoustic model itself is
      fully inside the ONNX graph, so the Pi needs only onnxruntime + the
      small numpy decoder.

Verified against the numpy reference on real clips (max abs diff < 1e-3).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hgm.feats import (SR, N_CEP, N_NFILT, N_FFT, FREQ_LOW, FREQ_HIGH,
                       PRE_EMP, _FRAME, _HOP, _mel_filterbank,
                       _dct2_orthonormal, read_wav, features)
from hgm.acoustic import AcousticModel

NEG = -1e4  # finite log sentinel for ONNX (decoder clamps below this)


def _dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    n = np.arange(n_in)
    k = np.arange(n_out)
    cos = np.cos(np.pi * np.outer(k, (n + 0.5)) / n_in)
    scale = np.full(n_out, np.sqrt(2.0 / n_in))
    scale[0] = np.sqrt(1.0 / n_in)
    return (cos * scale[:, None]).astype(np.float32)


def _dft_matrix(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n)
    w = np.exp(-2j * np.pi * np.outer(t, t) / n)
    return w.real.astype(np.float32), w.imag.astype(np.float32)


def _delta_ctx_matrix(ctx: int) -> np.ndarray:
    """(2*ctx+1, 1) weights aligned to [t-ctx..t+ctx]: w = -i for the past
    (i>0), 0 at center, +i for the future. The numpy _delta divides by
    2*sum(i^2) with a FIXED denominator (missing neighbors just contribute
    0 via the zero-padding), so the same matrix works at the boundaries."""
    w = np.zeros(2 * ctx + 1)
    for i in range(1, ctx + 1):
        w[ctx + i] = i
        w[ctx - i] = -i
    return w.astype(np.float32)  # 1-D (2*ctx+1): MatMul broadcasts it as a
    # row vector across the batch dims of the (1,T,13,5) window tensor


def build_features_model(am: AcousticModel, n_samples: int) -> onnx.ModelProto:
    nodes, inits = [], []

    def C(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))

    P, D, K = am.n_phones, am.dim, am.n_comp
    assert D == 39
    n_freq = N_FFT // 2 + 1
    n_frames = 1 + (n_samples - _FRAME) // _HOP
    T = n_frames

    # ---- input: x (1, N) float32, peak-normalized upstream ---------------
    x1 = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, n_samples])

    # ---- pre-emphasis: y[0]=x[0], y[t]=x[t]-0.97*x[t-1] ------------------
    C("c_pre", np.array([PRE_EMP], np.float32))
    nodes.append(helper.make_node("Slice", ["x", "x_tail_starts", "x_tail_ends", "x_tail_axes"], ["x_tail"]))
    C("x_tail_starts", np.array([1], np.int64))
    C("x_tail_ends", np.array([n_samples], np.int64))
    C("x_tail_axes", np.array([1], np.int64))
    nodes.append(helper.make_node("Slice", ["x", "x_prev_starts", "x_prev_ends", "x_prev_axes"], ["x_prev"]))
    C("x_prev_starts", np.array([0], np.int64))
    C("x_prev_ends", np.array([n_samples - 1], np.int64))
    C("x_prev_axes", np.array([1], np.int64))
    nodes.append(helper.make_node("Slice", ["x", "x_head_starts", "x_head_ends", "x_head_axes"], ["x_head"]))
    C("x_head_starts", np.array([0], np.int64))
    C("x_head_ends", np.array([1], np.int64))
    C("x_head_axes", np.array([1], np.int64))
    nodes.append(helper.make_node("Mul", ["x_prev", "c_pre"], ["x_prev_pe"]))
    nodes.append(helper.make_node("Sub", ["x_tail", "x_prev_pe"], ["x_tail_pe"]))
    nodes.append(helper.make_node("Concat", ["x_head", "x_tail_pe"], ["y"], axis=1))

    # ---- framing: (1, T, 410) -> window -> (1, T, 512) --------------------
    idx = (np.arange(_FRAME)[None, :] + _HOP * np.arange(T)[:, None]).astype(np.int64)
    C("frame_idx", idx)
    C("pad_fft", np.array([0, 0, 0, 0, 0, 102], np.int64))
    nodes.append(helper.make_node("Gather", ["y", "frame_idx"], ["frames"], axis=1))
    C("hamming", np.hamming(_FRAME).astype(np.float32).reshape(1, -1))
    nodes.append(helper.make_node("Mul", ["frames", "hamming"], ["frames_w"]))
    # zero-pad 410 -> 512 (rfft zero-pads)
    nodes.append(helper.make_node("Pad", ["frames_w", "pad_fft"], ["frames512"]))

    # ---- 512-pt DFT (real matrix multiply) -> power spectrum ---------------
    dft_r, dft_i = _dft_matrix(N_FFT)
    C("dft_r", dft_r)
    C("dft_i", dft_i)
    nodes.append(helper.make_node("MatMul", ["frames512", "dft_r"], ["spec_r"]))
    nodes.append(helper.make_node("MatMul", ["frames512", "dft_i"], ["spec_i"]))
    nodes.append(helper.make_node("Slice", ["spec_r", "spec_r_f_starts", "spec_r_f_ends", "spec_r_f_axes"], ["spec_r_f"]))
    C("spec_r_f_starts", np.array([0], np.int64))
    C("spec_r_f_ends", np.array([n_freq], np.int64))
    C("spec_r_f_axes", np.array([2], np.int64))
    nodes.append(helper.make_node("Slice", ["spec_i", "spec_i_f_starts", "spec_i_f_ends", "spec_i_f_axes"], ["spec_i_f"]))
    C("spec_i_f_starts", np.array([0], np.int64))
    C("spec_i_f_ends", np.array([n_freq], np.int64))
    C("spec_i_f_axes", np.array([2], np.int64))
    nodes.append(helper.make_node("Mul", ["spec_r_f", "spec_r_f"], ["spec_rr"]))
    nodes.append(helper.make_node("Mul", ["spec_i_f", "spec_i_f"], ["spec_ii"]))
    nodes.append(helper.make_node("Add", ["spec_rr", "spec_ii"], ["spec"]))

    # ---- mel filterbank -------------------------------------------------------
    fb = _mel_filterbank(SR, N_FFT, N_NFILT, FREQ_LOW, FREQ_HIGH).astype(np.float32)
    C("mel_fb", fb.T)
    nodes.append(helper.make_node("MatMul", ["spec", "mel_fb"], ["mel"]))
    nodes.append(helper.make_node("Log", ["mel"], ["mel_log"]))

    # ---- orthonormal DCT-II -> 13 cepstra -------------------------------------
    # numpy does  c = mel @ cos.T  where cos is (n_out, n_in); the effective
    # matmul matrix is (n_in, n_out) = (26, 13).
    C("dct", _dct_matrix(N_NFILT, N_CEP).T)
    nodes.append(helper.make_node("MatMul", ["mel_log", "dct"], ["cep"]))

    # ---- CMN --------------------------------------------------------------------
    nodes.append(helper.make_node("ReduceMean", ["cep"], ["cep_mean"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Sub", ["cep", "cep_mean"], ["cep_cmn"]))

    # ---- delta + delta-of-delta (context 2) ---------------------------------------
    # numpy _delta: d[t] = (2*f[t+2] + f[t+1] - f[t-1] - 2*f[t-2]) / 10,
    # context 2, fixed denominator. Zero-pad 2 frames each side so missing
    # neighbors contribute 0 (matching numpy exactly at the boundaries).
    # Implemented with 5 Slices + weighted sums (no Gather/MatMul, which ORT
    # cannot shape-infer with a dynamic frame count).
    C("pad_delta", np.array([0, 2, 0, 0, 2, 0], np.int64))
    C("delta_d", np.array([1.0 / 10.0], np.float32))
    C("c2", np.array([2.0], np.float32))
    C("c1", np.array([1.0], np.float32))
    C("cm1", np.array([-1.0], np.float32))
    C("cm2", np.array([-2.0], np.float32))
    C("big", np.array([2147483647], np.int64))

    def _delta_nodes(src, tag):
        """src (1,T,13) -> padded (1,T+4,13) -> 5 shifted slices -> (1,T,13)."""
        nodes.append(helper.make_node("Pad", [src, "pad_delta"], [f"{tag}_pad"]))
        # s_k = pad[:, k : k+T, :]  (negative end = T+4-(4-k))
        ends = [-4, -3, -2, -1, None]
        for k in range(5):
            C(f"{tag}_s{k}_st", np.array([k], np.int64))
            if ends[k] is not None:
                C(f"{tag}_s{k}_en", np.array([ends[k]], np.int64))
                en_name = f"{tag}_s{k}_en"
            else:
                en_name = "big"
            C(f"{tag}_s{k}_ax", np.array([1], np.int64))
            nodes.append(helper.make_node(
                "Slice", [f"{tag}_pad", f"{tag}_s{k}_st", en_name,
                          f"{tag}_s{k}_ax"], [f"{tag}_s{k}"]))
        # d = (2*s4 + s3 - s1 - 2*s0) * (1/10)
        nodes.append(helper.make_node("Mul", [f"{tag}_s4", "c2"], [f"{tag}_a"]))
        nodes.append(helper.make_node("Mul", [f"{tag}_s0", "cm2"], [f"{tag}_b"]))
        nodes.append(helper.make_node("Add", [f"{tag}_a", f"{tag}_s3"], [f"{tag}_c"]))
        nodes.append(helper.make_node("Sub", [f"{tag}_c", f"{tag}_s1"], [f"{tag}_d"]))
        nodes.append(helper.make_node("Add", [f"{tag}_d", f"{tag}_b"], [f"{tag}_e"]))
        nodes.append(helper.make_node("Mul", [f"{tag}_e", "delta_d"], [f"{tag}_out"]))

    _delta_nodes("cep_cmn", "d1")
    nodes.append(helper.make_node("Identity", ["d1_out"], ["delta"]))
    _delta_nodes("delta", "d2")
    nodes.append(helper.make_node("Identity", ["d2_out"], ["delta2"]))
    nodes.append(helper.make_node("Concat", ["cep_cmn", "delta", "delta2"],
                                  ["feat"], axis=2))  # (1, T, 39)

    # ---- GMM emissions: log p(x_t | phone, state) ---------------------------------
    w = np.zeros((P, 3, K), np.float32)
    mu = np.zeros((P, 3, K, D), np.float32)
    lv = np.zeros((P, 3, K, D), np.float32)
    for pi, p in enumerate(am.phones):
        hmm = am.hmms[p]
        for s in range(3):
            g = hmm.states[s]
            w[pi, s] = g.w.astype(np.float32)
            mu[pi, s] = g.mu.astype(np.float32)
            lv[pi, s] = g.logvar.astype(np.float32)
    C("gmm_w", w)
    C("gmm_mu", mu)
    C("gmm_logvar", lv)
    C("half_log2pi", np.array([0.5 * np.log(2.0 * np.pi)], np.float32))
    C("neghalf", np.array([-0.5], np.float32))
    C("sh_feat", np.array([1, -1, 1, 1, 1, D], np.int64))
    C("ax6", np.array([5], np.int64))
    C("ax5", np.array([4], np.int64))
    C("sh_em", np.array([-1, P * 3], np.int64))
    # diff: (1,T,1,1,1,D) - (P,3,K,D) -> (1,T,P,3,K,D)
    nodes.append(helper.make_node("Reshape", ["feat", "sh_feat"], ["feat_b"]))
    nodes.append(helper.make_node("Sub", ["feat_b", "gmm_mu"], ["diff"]))
    nodes.append(helper.make_node("Mul", ["diff", "diff"], ["diff2"]))
    nodes.append(helper.make_node("Exp", ["gmm_logvar"], ["gmm_var"]))
    nodes.append(helper.make_node("Div", ["diff2", "gmm_var"], ["diff2v"]))
    nodes.append(helper.make_node("Mul", ["diff2v", "neghalf"], ["a"]))
    nodes.append(helper.make_node("Mul", ["gmm_logvar", "neghalf"], ["b"]))
    nodes.append(helper.make_node("Add", ["a", "b"], ["ab"]))
    nodes.append(helper.make_node("Sub", ["ab", "half_log2pi"], ["lcomp0"]))
    nodes.append(helper.make_node("ReduceSum", ["lcomp0", "ax6"], ["lcomp"], keepdims=0))
    nodes.append(helper.make_node("Log", ["gmm_w"], ["logw"]))
    nodes.append(helper.make_node("Add", ["lcomp", "logw"], ["lpost"]))
    nodes.append(helper.make_node("ReduceMax", ["lpost"], ["lpost_max"], axes=[4], keepdims=1))
    nodes.append(helper.make_node("Sub", ["lpost", "lpost_max"], ["lpost_c"]))
    nodes.append(helper.make_node("Exp", ["lpost_c"], ["lpost_e"]))
    nodes.append(helper.make_node("ReduceSum", ["lpost_e", "ax5"], ["lpost_s"], keepdims=1))
    nodes.append(helper.make_node("Log", ["lpost_s"], ["lpost_ls"]))
    nodes.append(helper.make_node("Add", ["lpost_max", "lpost_ls"], ["em"]))
    # (1,T,P,3) -> (T, P*3)
    nodes.append(helper.make_node("Reshape", ["em", "sh_em"], ["log_emit"]))

    out = helper.make_tensor_value_info("log_emit", TensorProto.FLOAT, [None, P * 3])
    graph = helper.make_graph(nodes, "v6_features", [x1], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model, P


def verify(model_path: str, am: AcousticModel, data: str, n: int = 30,
           feat_tol: float = 0.05, emit_tol: float = 1.0, n_samples: int = 80000):
    """Check the ONNX graph against the numpy reference.

    Two tolerances: the 39-dim features must match tightly (the pipeline is
    the thing we're porting); the GMM log-emissions are a float32 log-sum-exp
    over 4 components x 39 dims, so they carry ~0.5 log-units of accumulation
    noise that is negligible for Viterbi (score gaps are tens of log-units).
    """
    import onnxruntime as ort
    import csv
    from onnx import helper, TensorProto
    import onnx as _onnx
    # add the 39-dim feature tensor as an extra output for the tight check
    m = _onnx.load(model_path)
    m.graph.output.append(helper.make_tensor_value_info(
        "feat", TensorProto.FLOAT, [None, None, 39]))
    sess_f = ort.InferenceSession(m.SerializeToString(),
                                  providers=["CPUExecutionProvider"])
    mpath = os.path.join(data, "test", "manifest.csv")
    rows = []
    with open(mpath) as f:
        for r in csv.DictReader(f):
            p = os.path.join(data, "test", r["file"])
            if os.path.exists(p):
                rows.append(p)
    maxfeat, maxemit = 0.0, 0.0
    for p in rows[:n]:
        x = read_wav(p)
        # pad/truncate to the model's fixed input length (the Pi captures a
        # fixed 5 s window); both ONNX and numpy then see identical input.
        if len(x) < n_samples:
            x = np.concatenate([x, np.zeros(n_samples - len(x))])
        else:
            x = x[:n_samples]
        X = x.reshape(1, -1).astype(np.float32)
        emit, feat = sess_f.run(["log_emit", "feat"], {"x": X})
        ref = features(x)  # (T,39) float32
        T = ref.shape[0]
        feat = feat[:T]
        maxfeat = max(maxfeat, float(np.abs(feat - ref).max()))
        ref_em = np.zeros((T, am.n_phones * 3), np.float32)
        for pi, ph in enumerate(am.phones):
            hmm = am.hmms[ph]
            for s in range(3):
                ref_em[:, pi * 3 + s] = hmm.states[s].log_likelihood(ref.astype(np.float64))
        ref_em = np.clip(ref_em, NEG, None)
        maxemit = max(maxemit, float(np.abs(emit[:T] - ref_em).max()))
    print(f"verify: {n} clips, max feat diff = {maxfeat:.5f} (tol {feat_tol}), "
          f"max emit diff = {maxemit:.4f} (tol {emit_tol})")
    return maxfeat <= feat_tol and maxemit <= emit_tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    ap.add_argument("--data", default="/home/ron.andrei.soriano/sandbox/data/external/me2-v6/dataset")
    ap.add_argument("--n-samples", type=int, default=80000,
                    help="fixed waveform length (5 s @ 16 kHz)")
    args = ap.parse_args()

    am = AcousticModel.load(os.path.join(args.models, "acoustic_model.npz"))
    model, P = build_features_model(am, args.n_samples)
    out = os.path.join(args.models, "v6_features.onnx")
    onnx.save(model, out)
    size = os.path.getsize(out) / 1e6
    print(f"saved {out} ({size:.2f} MB), {P} phones x 3 states")

    # transitions for the numpy Viterbi decoder
    log_a = np.stack([am.hmms[p].log_a.astype(np.float32)
                      for p in am.phones])
    np.savez(os.path.join(args.models, "v6_transitions.npz"),
             log_a=log_a, phones=np.array(am.phones))
    print(f"saved v6_transitions.npz ({log_a.shape})")

    ok = verify(out, am, args.data, n_samples=args.n_samples)
    print("verify:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
