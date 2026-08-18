import time, torch
print("  torch      :", torch.__version__, "| cuda", torch.version.cuda)
print("  available  :", torch.cuda.is_available())
print("  device     :", torch.cuda.get_device_name(0))
print("  capability : sm_%d%d" % torch.cuda.get_device_capability(0))
print("  arch list  :", torch.cuda.get_arch_list())
for m in ["mmcv", "mmdet", "mmseg", "mmdet3d"]:
    mod = __import__(m)
    print("  %-10s:" % m, getattr(mod, "__version__", "?"))

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction as F
v = torch.rand(1, 100, 8, 32, device="cuda")
s = torch.tensor([[10, 10]], dtype=torch.long, device="cuda")
l = torch.tensor([0], dtype=torch.long, device="cuda")
loc = torch.rand(1, 16, 8, 1, 4, 2, device="cuda")
a = torch.rand(1, 16, 8, 1, 4, device="cuda"); a = a / a.sum(-1, keepdim=True)
t0 = time.time(); out = F.apply(v, s, l, loc, a, 64); torch.cuda.synchronize()
print("  ms_deform_attn : %s finite=%s (%.2fs)" %
      (tuple(out.shape), bool(torch.isfinite(out).all()), time.time() - t0))

import importlib
from mmcv import Config
cfg = Config.fromfile("projects/configs/VAD/VAD_etri_tiny.py")
importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
from mmdet3d.models import build_model
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")).cuda().eval()
n = sum(p.numel() for p in model.parameters())
x = torch.randn(6, 3, 448, 768, device="cuda")
with torch.no_grad():
    f = model.img_neck(model.img_backbone(x))
torch.cuda.synchronize()
print("  VAD model      : %.2f M params, neck %s, peak %.2f GB"
      % (n / 1e6, tuple(f[0].shape), torch.cuda.max_memory_allocated() / 1e9))
print("  ALL STACK CHECKS PASSED")
