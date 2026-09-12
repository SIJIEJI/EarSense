import argparse
import numpy as np

def aggregate(X, mask, k=5, mode="mean"):
    """
    X: (N,T,F) at 1-min resolution
    mask: (N,T)
    return X2: (N,T2,F), mask2: (N,T2)
    """
    N, T, F = X.shape
    T2 = (T + k - 1) // k
    X2 = np.zeros((N, T2, F), dtype=np.float32)
    m2 = np.zeros((N, T2), dtype=np.float32)

    for i in range(T2):
        s = i * k
        e = min((i + 1) * k, T)
        xm = X[:, s:e, :]
        mm = mask[:, s:e]

        valid = (mm > 0.5)
        m2[:, i] = (valid.sum(axis=1) > 0).astype(np.float32)

        if mode == "mean":
            # masked mean
            denom = np.maximum(valid.sum(axis=1, keepdims=True), 1)
            X2[:, i, :] = (xm * valid[..., None]).sum(axis=1) / denom
        elif mode == "mean_std":
            denom = np.maximum(valid.sum(axis=1, keepdims=True), 1)
            mu = (xm * valid[..., None]).sum(axis=1) / denom
            # masked var
            v = ((xm - mu[:, None, :]) ** 2) * valid[..., None]
            var = v.sum(axis=1) / denom
            sd = np.sqrt(np.maximum(var, 1e-6))
            X2[:, i, :] = mu  # if you want concat, do it below
            # if you want concat mean+std:
            # X2 = np.concatenate([mu, sd], axis=-1)  # (N,T2,2F)
        else:
            raise ValueError("mode must be mean or mean_std")

    return X2, m2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=False, default="./nn_dataset_subject_level.npz")
    ap.add_argument("--out_npz", required=False, default="./nn_dataset_subject_level_5min.npz")
    ap.add_argument("--k", type=int, default=5, help="minutes to aggregate")
    ap.add_argument("--mode", choices=["mean"], default="mean")
    args = ap.parse_args()

    d = np.load(args.in_npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    mask = d["mask"].astype(np.float32)

    X2, m2 = aggregate(X, mask, k=args.k, mode=args.mode)

    out = dict(d)
    out["X"] = X2
    out["mask"] = m2
    np.savez(args.out_npz, **out)

    print("Saved:", args.out_npz)
    print("X:", X2.shape, "mask:", m2.shape)

if __name__ == "__main__":
    main()
