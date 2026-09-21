"""꼬리 최종 Linear 의 출력을 fut_ts(6) 스텝에서 T(10) 스텝으로 넓힌 체크포인트를 만든다.

앞 6 스텝의 가중치는 그대로 복사하고, 7~T 스텝은 6번째 스텝의 가중치를 복제한다(초기 예측이
마지막 속도를 유지하며 이어지도록). 나머지 키는 전부 그대로.

  python tools/expand_fut_ckpt.py <입력.pth> <출력.pth> [--modes 3] [--fut-ts 6] [--out-ts 10]
"""
import argparse
import torch

p = argparse.ArgumentParser()
p.add_argument('src')
p.add_argument('dst')
p.add_argument('--modes', type=int, default=3)
p.add_argument('--fut-ts', type=int, default=6)
p.add_argument('--out-ts', type=int, default=10)
a = p.parse_args()
ck = torch.load(a.src, map_location='cpu')
sd = ck['state_dict'] if 'state_dict' in ck else ck
M, F, T = a.modes, a.fut_ts, a.out_ts
old_n = M * F * 2
keys = [k for k in sd if k.startswith('pts_bbox_head.ego_fut_decoder.') and sd[k].shape[0] == old_n]
assert len(keys) == 2, keys   # weight, bias
for k in keys:
    v = sd[k]
    if v.dim() == 2:
        new = v.new_zeros(M * T * 2, v.shape[1])
        ov = v.view(M, F, 2, -1)
        nv = new.view(M, T, 2, -1)
    else:
        new = v.new_zeros(M * T * 2)
        ov = v.view(M, F, 2)
        nv = new.view(M, T, 2)
    nv[:, :F] = ov
    nv[:, F:] = ov[:, F - 1:F].expand_as(nv[:, F:])
    sd[k] = new
    print('%s: %s -> %s' % (k, tuple(v.shape), tuple(new.shape)))
if 'state_dict' in ck:
    ck['state_dict'] = sd
meta = ck.get('meta', {}) or {}
meta['expanded_fut'] = dict(modes=M, fut_ts=F, out_ts=T)
ck['meta'] = meta
torch.save(ck, a.dst)
print('저장', a.dst)
