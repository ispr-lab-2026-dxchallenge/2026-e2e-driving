"""제출 프로토콜과 똑같이 돌리되 모드별 예측·GT·목표점·과거변위를 npz 로 덤프한다.

BEV/스트림 처리는 tools/eval_protocol.py 의 함수를 그대로 재사용한다(동일 경로 보장).
사용: python tools/dump_preds.py <config> <ckpt> <out.npz>
"""
import sys, importlib, numpy as np, torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('ep', 'tools/eval_protocol.py')
EP = _ilu.module_from_spec(_spec)
import sys as _s; _argv = _s.argv; _s.argv = ['eval_protocol.py', 'x', 'y']
_spec.loader.exec_module(EP)
_s.argv = _argv

cfg_path, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = Config.fromfile(cfg_path)
importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
cfg.data.test.test_mode = True
cfg.data.test.pop('samples_per_gpu', None)
cfg.data.test.pop('map_ann_file', None)
dataset = build_dataset(cfg.data.test)
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, ckpt, map_location='cpu', strict=False)
model.compute_planner_metric_stp3 = lambda *a, **k: {}
model = MMDataParallel(model.cuda(0), device_ids=[0]); model.eval()

grouped = {}
for gi, info in enumerate(dataset.data_infos):
    grouped.setdefault(info['scene_token'], []).append(gi)
groups = list(grouped.items())
EP.reset_stream(model.module)
for k in range(5):
    with torch.no_grad():
        model(return_loss=False, rescale=True, **collate([dataset[groups[0][1][k]]], samples_per_gpu=1))
torch.cuda.synchronize()

offs = [-10, -5, 0]
P, G, GOAL, HIS, CMD, LCF = [], [], [], [], [], []
for si, (_t, ids) in enumerate(groups):
    for cur in range(EP.HIS_FRAMES, len(ids), 10):
        batches = [collate([dataset[ids[cur + o]]], samples_per_gpu=1) for o in offs]
        EP.reset_stream(model.module)
        result = None
        for bi, c in enumerate(batches):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                if bi < len(batches) - 1: EP.history_bev_step(model, c)
                else: result = model(return_loss=False, rescale=True, **c)
        c = batches[-1]
        if not bool(np.asarray(EP.unwrap(c, 'fut_valid_flag')).reshape(-1)[0]): continue
        P.append(result[0]['pts_bbox']['ego_fut_preds'].cpu().double().numpy())
        G.append(np.asarray(EP.unwrap(c, 'ego_fut_trajs')).reshape(-1, 2))
        lcf = np.asarray(EP.unwrap(c, 'ego_lcf_feat')).reshape(-1)
        LCF.append(lcf)
        GOAL.append(lcf[11:13] if lcf.shape[0] >= 13 else lcf[9:11])
        HIS.append(np.asarray(EP.unwrap(c, 'ego_his_trajs')).reshape(-1, 2))
        CMD.append(np.asarray(EP.unwrap(c, 'ego_fut_cmd')).reshape(-1))
    print('  시나리오 %d/%d  누적 %d' % (si + 1, len(groups), len(P)), flush=True)
np.savez_compressed(out, pred=np.array(P), gt=np.array(G), goal=np.array(GOAL),
                    his=np.array(HIS), cmd=np.array(CMD), lcf=np.array(LCF))
print('저장', out, np.array(P).shape, np.array(HIS).shape)
