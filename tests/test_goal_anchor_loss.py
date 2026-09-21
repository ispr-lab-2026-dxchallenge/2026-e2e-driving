"""loss_goal_anchor 단위 검사 (GPU 불필요). 함수 본문만 뽑아 실행한다."""
import types
import torch

src = open('projects/mmdet3d_plugin/VAD/VAD_head.py', encoding='utf-8').read()
beg = src.index('    def loss_goal_anchor(')
end = src.index('    def loss_anchor(')
code = 'import torch\n' + '\n'.join(l[4:] for l in src[beg:end].splitlines())
ns = {}
exec(code, ns)
fn = ns['loss_goal_anchor']
self = types.SimpleNamespace(fut_ts=6, goal_anchor_weight=1.0,
                             transformer=types.SimpleNamespace(last_goal_dropped=False))

B, M, T = 4, 3, 10
gt = torch.randn(B, 6, 2) * 0.5 + torch.tensor([5.0, 0.0])          # 3초 GT 변위
masks = torch.ones(B, 6)
cmd = torch.zeros(B, M); cmd[:, 2] = 1
gt3 = gt.cumsum(1)[:, -1]
goal = gt3 + torch.tensor([20.0, -2.0])                            # 5초 목표점
lcf = torch.zeros(B, 1, 13); lcf[:, 0, 9:11] = goal

# 완벽한 예측: 앞 6 = GT, 뒤 4 = (goal-gt3)/4 → 손실 0
ext = torch.zeros(B, M, T, 2)
ext[:, :, :6] = gt[:, None]
ext[:, :, 6:] = ((goal - gt3) / 4)[:, None, None]
l0 = fn(self, ext, gt, masks, cmd, lcf)
assert l0.item() < 1e-6, l0
# 뒤 스텝을 흔들면 손실 > 0
e2 = ext.clone(); e2[:, 2, 7] += 1.0
l2 = fn(self, e2, gt, masks, cmd, lcf); assert l2.item() > 0
# 비선택 모드를 흔들면 불변
e3 = ext.clone(); e3[:, 0] += 100.0
l3 = fn(self, e3, gt, masks, cmd, lcf); assert l3.item() < 1e-6
# 3초 궤적이 드리프트하면(앞 6 스텝 각 +0.1) 끝점도 밀리므로 손실 > 0 — 앵커가 3초 눈금을 묶는다
e4 = ext.clone(); e4[:, :, :6] += 0.1
l4 = fn(self, e4, gt, masks, cmd, lcf); assert l4.item() > 0
# 무효 표본(마스크 0)은 기여 없음
m5 = masks.clone(); m5[:, -1] = 0
l5 = fn(self, e2, gt, m5, cmd, lcf); assert l5.item() < 1e-6
# condition dropout 으로 목표점이 지워진 배치는 앵커를 걸지 않는다
self.transformer.last_goal_dropped = True
l6 = fn(self, e2, gt, masks, cmd, lcf); assert l6.item() == 0.0
self.transformer.last_goal_dropped = False
# 기울기가 앞 6 스텝까지 흐른다
e7 = e4.clone().requires_grad_(True)
fn(self, e7, gt, masks, cmd, lcf).backward()
assert e7.grad[:, 2, :6].abs().sum() > 0
print('단위 검사 PASS  (완벽 %.2e / 뒤스텝 교란 %.3f / 비선택모드 %.2e / 드리프트 %.3f / 마스크 %.2e / 드롭배치 %.1f)'
      % (l0, l2, l3, l4, l5, l6))
