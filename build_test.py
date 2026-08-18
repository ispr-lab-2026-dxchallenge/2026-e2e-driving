import importlib, time, torch
from mmcv import Config

CFG = "projects/configs/VAD/VAD_etri_tiny.py"
print("=" * 62)
cfg = Config.fromfile(CFG)
print("config loaded :", CFG)
print("  bev          : %dx%d" % (cfg.bev_h_, cfg.bev_w_))
print("  queue_length :", cfg.queue_length)
print("  epochs       :", cfg.total_epochs)
print("  dataset type :", cfg.dataset_type)

importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
print("plugin imported:", cfg.plugin_dir)

from projects.mmdet3d_plugin.datasets.etri_vad_dataset import VADCustomETRIDataset
print("ETRI dataset class OK :", VADCustomETRIDataset.__name__)
import tools.data_converter.etri_vad_converter as _c
print("train converter import OK")
import tools.data_converter.etri_test_converter as _t
print("test converter import OK")
print("-" * 62)

from mmdet3d.models import build_model
t0 = time.time()
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
print("model built in %.1fs : %s" % (time.time() - t0, type(model).__name__))
n = sum(p.numel() for p in model.parameters())
nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("params        : %.2f M  (trainable %.2f M)" % (n / 1e6, nt / 1e6))
print("head          :", type(model.pts_bbox_head).__name__)
print("transformer   :", type(model.pts_bbox_head.transformer).__name__)
print("-" * 62)

model = model.cuda().eval()
print("moved to cuda OK")
x = torch.randn(6, 3, 448, 768, device="cuda")   # 6 cams, tiny scale
with torch.no_grad():
    feats = model.img_backbone(x)
    necks = model.img_neck(feats)
torch.cuda.synchronize()
print("backbone+neck forward OK")
for i, f in enumerate(necks):
    print("  neck[%d] : %s" % (i, tuple(f.shape)))
print("peak VRAM     : %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("=" * 62)
print("BUILD TEST PASSED")
