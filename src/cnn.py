
"""CNN training code. All model decisions use the VALIDATION set only."""
import math, time, random, json
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
import preprocessing as pp, evaluation as ev

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

def load_images(master, proc_root):
    """Load every processed 512-px image once into memory: {image_id: uint8 RGB array}."""
    return {r.image_id: pp.load_processed(proc_root / r.processed_rel) for r in master.itertuples()}

class FundusDataset(Dataset):
    def __init__(self, ids, labels, images, version, train):
        self.ids, self.labels, self.images = list(ids), np.asarray(labels), images
        self.version, self.train = version, train
        self.aug = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                              T.RandomAffine(degrees=180, scale=(0.95, 1.05)),
                              T.ColorJitter(brightness=0.1, contrast=0.1)])
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        img = self.images[self.ids[i]]
        if self.version == 'B':
            img = pp.version_b(img)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        if self.train:
            x = self.aug(x)
        x = (x.float() / 255.0 - MEAN) / STD
        return x, int(self.labels[i])

def build_model(arch="resnet18"):
    if arch != "resnet18":
        raise ValueError(arch)
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 5)
    return m

def _loader(ds, batch, shuffle, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2, pin_memory=True,
                      drop_last=shuffle, generator=g,
                      worker_init_fn=lambda w: np.random.seed(seed + w))

@torch.no_grad()
def predict(model, ids, labels, images, version, batch=64, device="cuda"):
    model.eval()
    dl = _loader(FundusDataset(ids, labels, images, version, train=False), batch, False, 0)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(x.to(device, non_blocking=True)).float(), 1).cpu())
    return torch.cat(out).numpy()

def train_run(cfg, master, images, models_dir, logs_dir, device="cuda"):
    set_seed(cfg["seed"])
    tr, va = master[master.set == "train"], master[master.set == "val"]
    dl_tr = _loader(FundusDataset(tr.image_id, tr.dr_grade, images, cfg["version"], True),
                    cfg["batch"], True, cfg["seed"])
    model = build_model(cfg["arch"]).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps, warm = cfg["epochs"] * len(dl_tr), cfg["warmup"] * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda")
    best, best_epoch, wait, log = -1.0, 0, 0, []
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        prob = predict(model, va.image_id, va.dr_grade, images, cfg["version"], device=device)
        yv = va.dr_grade.values
        vloss = crit(torch.log(torch.tensor(prob).clamp_min(1e-7)).to(device),
                     torch.tensor(yv, device=device)).item()   # weighted NLL on probabilities
        m = ev.compute_metrics(yv, prob, ev.threshold_for_sensitivity(yv, prob, 0.90))
        log.append({"epoch": epoch, "train_loss": tl / n, "val_loss": vloss, "val_QWK": m["QWK"],
                    "val_macro_F1": m["macro_F1"], "val_referable_AUROC": m["referable_AUROC"],
                    "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch:2d} | train loss {tl / n:.3f} | val loss {vloss:.3f} | "
              f"val QWK {m['QWK']:.3f} | macro-F1 {m['macro_F1']:.3f} | {(time.time() - t0) / 60:.1f} min")
        if m["QWK"] > best:
            best, best_epoch, wait = m["QWK"], epoch, 0
            torch.save(model.state_dict(), models_dir / f"{cfg['name']}.pt")
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"early stopping (no improvement for {cfg['patience']} epochs)"); break
    log = pd.DataFrame(log); log["best_epoch"] = best_epoch
    log.to_csv(logs_dir / f"{cfg['name']}_log.csv", index=False)
    json.dump({**cfg, "best_epoch": best_epoch, "best_val_QWK": best}, open(logs_dir / f"{cfg['name']}_config.json", "w"), indent=2)
    model.load_state_dict(torch.load(models_dir / f"{cfg['name']}.pt"))
    return model, log
