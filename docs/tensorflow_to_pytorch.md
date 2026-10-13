# From TensorFlow/Keras to PyTorch, for this project

The old experiments in this repo (`atlantic_exps2`, `atlantic_exps3`, now on the
`legacy-2024` branch) used TensorFlow/Keras. The new CNN in `src/tropcyc/cnn.py` is PyTorch.
This guide only covers what you need to read and change that one file, set next to the
Keras code you already know.

## The short version

| Idea | Keras | PyTorch (here) |
|---|---|---|
| Define a model | `tf.keras.Model` / functional API | subclass `nn.Module`, write `forward()` |
| Image layout | channels **last**: `(N, H, W, C)` | channels **first**: `(N, C, H, W)` |
| Feed data | NumPy arrays or `tf.data.Dataset` | `Dataset` + `DataLoader` |
| Train | `model.compile(...)` + `model.fit(...)` | a `for` loop you write (about 10 lines) |
| Early stopping | `callbacks.EarlyStopping(restore_best_weights=True)` | a few lines in the loop |
| Train vs inference mode | handled inside `fit` / `predict` | `model.train()` / `model.eval()` yourself |
| No gradients at inference | automatic in `predict` | `with torch.no_grad():` (or `@torch.no_grad()`) |
| Save | `model.save("m.keras")` | `torch.save(model.state_dict(), "m.pt")` |
| GPU | automatic | move the model and each batch with `.to(device)` |

The extra lines PyTorch makes you write are the trade: everything in the training step is
plain Python you can print, debug and change.

## 1. The model

The network has two inputs: an ERA5 patch `(5, 41, 41)` and 14 track features. Its one
output is the 24 h intensity change.

**Keras (functional API), which is how you would have written it:**

```python
from tensorflow.keras import layers, Model, Input

def block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.MaxPooling2D(2)(x)

patch = Input(shape=(41, 41, 5))            # channels LAST
feats = Input(shape=(14,))

x = block(patch, 32)
x = block(x, 64)
x = block(x, 128)
x = layers.GlobalAveragePooling2D()(x)

f = layers.Dense(64, activation="relu")(feats)

z = layers.Concatenate()([x, f])
z = layers.Dense(64, activation="relu")(z)
z = layers.Dropout(0.3)(z)
out = layers.Dense(1)(z)

model = Model([patch, feats], out)
```

**PyTorch (`IntensityNet` in `cnn.py`):**

```python
class IntensityNet(nn.Module):
    def __init__(self, n_channels, n_features, width=32, dropout=0.3):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.conv = nn.Sequential(block(n_channels, width), block(width, 2 * width),
                                  block(2 * width, 4 * width),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.track = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(4 * width + 64, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, patch, features):
        z = torch.cat([self.conv(patch), self.track(features)], dim=1)
        return self.head(z).squeeze(1)
```

What changes:

- **Layers are created in `__init__` and connected in `forward`.** Keras does both in one
  line. In PyTorch, `forward` is ordinary Python, so you can put `print(x.shape)` in it
  while debugging.
- **You give input sizes.** `nn.Conv2d(c_in, c_out, 3)` needs `c_in`, and `nn.Linear(64, 1)`
  needs 64, where Keras infers them. `4 * width + 64` is the length after the
  concatenation: 128 from the conv branch plus 64 from the track branch.
- **`padding=1` with a 3×3 kernel is `padding="same"`.**
- **`GlobalAveragePooling2D()` is `AdaptiveAvgPool2d(1)` + `Flatten()`.**
- **`Dense` is `Linear`, and activations are separate layers** (`nn.ReLU()`) instead of
  `activation="relu"`.
- **`.squeeze(1)`** turns the `(N, 1)` output into `(N,)` so it matches the target shape.
  Keras broadcasts quietly; PyTorch would broadcast `(N, 1)` against `(N,)` into `(N, N)`
  and train on the wrong loss without any error. Watch for this one.

## 2. Channels first

Your old arrays (for example `final_arrv3.npy`) were `(N, H, W, C)`, the Keras default.
PyTorch convolutions want `(N, C, H, W)`. `era5.py` already writes patches as `(C, H, W)`.
To reuse an old channels-last array:

```python
x = np.load("final_arrv3.npy")        # (N, H, W, C)
x = np.transpose(x, (0, 3, 1, 2))     # (N, C, H, W)
```

Getting this wrong usually doesn't crash. With 41×41 patches and 5 channels the first
conv fails, but a square array whose channel count equals its height would train quietly
on scrambled data.

## 3. Feeding data

In the old scripts you built `x_train` and `y_train` NumPy arrays and called
`model.fit(x_train, y_train, batch_size=64)`. PyTorch splits this into two pieces:

- a **`Dataset`** (`PatchDataset`) that knows how many samples there are (`__len__`) and how
  to return one (`__getitem__`). It is the equivalent of indexing your NumPy arrays.
- a **`DataLoader`** that batches and shuffles. It is the equivalent of `batch_size=` and
  `shuffle=True` in `fit`.

```python
train_dl = DataLoader(PatchDataset(...), batch_size=64, shuffle=True)
for x, f, y in train_dl:      # one batch of patches, features, targets
    ...
```

Normalisation happens in `PatchDataset` using statistics from the **training storms
only**, the same as fitting a `Normalization` layer with `.adapt()` on training data.
Missing values (SST over land) become 0 after normalising, which is the channel mean.

## 4. Training: `compile` + `fit` written out

Keras:

```python
model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3),
              loss=tf.keras.losses.Huber(delta=10.0))
model.fit([x_tr, f_tr], y_tr, validation_data=([x_va, f_va], y_va), epochs=40,
          callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])
```

PyTorch (`CNNModel.fit`, trimmed):

```python
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.HuberLoss(delta=10.0)

for epoch in range(40):
    net.train()                      # dropout on, batch norm uses batch stats
    for x, f, y in train_dl:
        opt.zero_grad()              # clear gradients from the last step
        loss = loss_fn(net(x, f), y) # forward pass
        loss.backward()              # backpropagation
        opt.step()                   # update weights
    val_mae = evaluate(val_dl)       # uses net.eval() + torch.no_grad()
    # early stopping, by hand:
    if val_mae < best:
        best, best_state, bad = val_mae, copy.deepcopy(net.state_dict()), 0
    else:
        bad += 1
        if bad >= 6:
            break
net.load_state_dict(best_state)      # = restore_best_weights=True
```

The four lines inside the batch loop are what `fit` does for you on every batch. Forgetting
`opt.zero_grad()` is the classic mistake: gradients then add up across batches.

## 5. `train()` and `eval()`

Keras switches dropout and batch normalisation automatically: training behaviour inside
`fit`, inference behaviour inside `predict`. In PyTorch you switch the mode yourself:

- `net.train()` before training batches,
- `net.eval()` before validation or prediction, together with `torch.no_grad()` so no
  gradient memory is used.

If you forget `eval()`, predictions come out noisy (dropout is still active) and depend on
the batch (batch norm uses batch statistics). `_predict_loader` in `cnn.py` does both.

## 6. Validation split

`model.fit(validation_split=0.15)` takes the last 15% of **rows**. For storms that leaks:
neighbouring 6 h fixes of the same storm end up on both sides. `CNNModel.fit` holds out
the most recent 15% of **storms** instead, and `evaluate.split_by_year` splits whole
seasons for the final test. Your old `test1_20y_RNN.py` already split by storm ID, so this
is the same idea.

## 7. Saving and loading

Keras saves the architecture and the weights together. PyTorch normally saves only the
weights (`state_dict`), so you rebuild the class and then load into it:

```python
torch.save({"state": net.state_dict(), "stats": stats}, "cnn.pt")   # CNNModel.save

ckpt = torch.load("cnn.pt", weights_only=False)
net = IntensityNet(n_channels=5, n_features=14)
net.load_state_dict(ckpt["state"])
net.eval()
```

The normalisation statistics are stored in the same file. Without them, new patches
can't be scaled the way the training data was.

## 8. GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
net = net.to(device)
x, f, y = x.to(device), f.to(device), y.to(device)   # every batch
```

The model and the data must be on the same device. "Expected all tensors to be on the
same device" means one `.to(device)` is missing.

## 9. Error messages you will probably meet

| Message | Usual cause here |
|---|---|
| `mat1 and mat2 shapes cannot be multiplied` | a `Linear` input size is wrong, e.g. after changing `width` or the patch size |
| `Expected ... input[64, 41, 41, 5] to have 5 channels` | channels-last array, see section 2 |
| `expected scalar type Float but found Double` | NumPy float64 into the model; use `.astype(np.float32)` |
| Loss is `nan` from the first step | NaNs in the inputs; check `np.isnan(x).any()` (SST over land) |
| Validation loss jumps around between runs | missing `net.eval()` before validating |

## Where to change things

- Patch size: `--half-width` in `tropcyc era5`. `AdaptiveAvgPool2d` means the network
  needs no change.
- Channels: `CHANNELS` and `extract_patch` in `era5.py`. The network reads the channel count
  from the data.
- Depth or width: `IntensityNet(width=...)`, or add a `block(...)`.
- Loss: `nn.HuberLoss(delta=10.0)`. Use `nn.MSELoss()` for plain MSE, or switch to an RI
  classifier with `nn.BCEWithLogitsLoss()` and the `ri_24` target.
