"""제출·재현 패키지용: 추론에 쓰이지 않는 가중치(VO 헤드 등)를 체크포인트에서 제거한다.

제출 config(exp_full_tgtbev)는 VO 모듈을 만들지 않아 로드 시 무시되지만, 심사자가 받은
파일에 쓰이지 않는 키가 남아 있으면 오해를 산다. 제거 전후로 추론 결과는 완전히 동일하다.

  python tools/strip_ckpt.py <입력.pth> <출력.pth> [--prefix vo_]
"""
import argparse, torch

p = argparse.ArgumentParser()
p.add_argument('src'); p.add_argument('dst')
p.add_argument('--prefix', nargs='*', default=['vo_'])
a = p.parse_args()

ck = torch.load(a.src, map_location='cpu')
sd = ck['state_dict'] if 'state_dict' in ck else ck
drop = [k for k in sd if any(pf in k.split('.')[-2] or k.split('.')[-1].startswith(pf) or ('.' + pf) in k for pf in a.prefix)]
drop = [k for k in sd if any(('.' + pf) in k or k.startswith(pf) for pf in a.prefix)]
n_before = len(sd)
for k in drop:
    del sd[k]
if 'state_dict' in ck:
    ck['state_dict'] = sd
meta = ck.get('meta', {}) or {}
meta['stripped'] = {'prefixes': a.prefix, 'removed': len(drop)}
ck['meta'] = meta
torch.save(ck, a.dst)
print('키 %d -> %d (제거 %d)' % (n_before, len(sd), len(drop)))
print('제거된 키 예시:', drop[:5])
