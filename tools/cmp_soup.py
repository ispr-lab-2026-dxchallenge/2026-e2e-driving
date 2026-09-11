"""두 체크포인트의 state_dict 가 동일한지 (메타 제외) 해시로 대조"""
import sys, hashlib, torch
def h(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    m = hashlib.md5()
    for k in sorted(sd):
        m.update(k.encode()); m.update(sd[k].contiguous().numpy().tobytes())
    return m.hexdigest(), len(sd)
for p in sys.argv[1:]:
    print(h(p), p)
