import torch


def random_crop(X, n_crop_frames):
    channel, dim, total_frames = X.shape
    n_crop_frames: int = n_crop_frames
    bgn_frame = torch.randint(low=0, high=total_frames - n_crop_frames, size=(1,))[0]
    X = X[:, :, bgn_frame : bgn_frame + n_crop_frames]
    return X


def make_subseq(X, n_crop_frames, n_hop_frames):
    # mel_spectrogram is np.ndarray [shape=(n_mels, t)]

    channel, dim, frames = X.shape

    total_frames = frames - n_crop_frames + 1

    # generate feature vectors by concatenating multiframes
    subseq = []
    for frame_idx in range(total_frames):
        subseq.append(X[:, :, frame_idx : frame_idx + n_crop_frames])

    subseq = torch.stack(subseq, dim=0)
    if not n_hop_frames == 0:
        subseq = subseq[::n_hop_frames, :, :]
    # subseq shape is (#batch, #channels, #dim, #frame)
    return subseq
