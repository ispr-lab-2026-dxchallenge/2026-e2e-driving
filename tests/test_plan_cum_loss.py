"""누적 좌표 손실(plan_reg_cum_weight) 단위 검사 — GPU 불필요."""
import torch
from mmdet.models.losses import L1Loss
loss_fn = L1Loss(loss_weight=1.0)
B, M, T = 2, 3, 6
gt = (torch.randn(B, 1, T, 2) * 0.1).repeat(1, M, 1, 1)
cmd = torch.zeros(B, M); cmd[:, 2] = 1
masks = torch.ones(B, T)
w = (cmd[..., None, None] * masks[:, None, :, None]).repeat(1, 1, 1, 2)
cw = torch.tensor([11., 11, 5, 5, 2, 2]); cw = cw * (cw.numel() / cw.sum())
cum_w = w * cw[None, None, :, None]

def losses(pred):
    disp = loss_fn(pred, gt, w)
    cum = loss_fn(torch.cumsum(pred, -2), torch.cumsum(gt, -2), cum_w)
    return disp.item(), cum.item()

# 1) 수동 계산과 일치
pred = gt + torch.randn_like(gt) * 0.05
_, cum = losses(pred)
manual = (torch.abs(torch.cumsum(pred, -2) - torch.cumsum(gt, -2)) * cum_w).mean().item()
assert abs(cum - manual) < 1e-6, (cum, manual)

# 2) 선택되지 않은 모드는 영향 없음
p2 = pred.clone(); p2[:, 0] += 100.0
assert abs(losses(p2)[1] - cum) < 1e-6

# 3) 완전 정답이면 둘 다 0
assert losses(gt) == (0.0, 0.0)

# 4) 같은 크기의 오차라도 '한 방향으로 계속 치우친 오차(드리프트)' 를 누적 손실이 훨씬 크게 벌한다
eps = 0.01
drift = gt + eps                                   # 매 스텝 같은 방향
alt = gt + eps * torch.tensor([1., -1, 1, -1, 1, -1])[None, None, :, None]   # 번갈아 상쇄
d_disp, d_cum = losses(drift); a_disp, a_cum = losses(alt)
print('변위 L1  드리프트 %.6f / 상쇄 %.6f  (비 %.2f)' % (d_disp, a_disp, d_disp / a_disp))
print('누적 L1  드리프트 %.6f / 상쇄 %.6f  (비 %.2f)' % (d_cum, a_cum, d_cum / a_cum))
assert abs(d_disp - a_disp) < 1e-6, '변위 손실은 두 경우를 구분하지 못한다'
assert d_cum > a_cum * 2, '누적 손실은 드리프트를 크게 벌해야 한다'
print('단위 검사 PASS (변위 손실은 드리프트와 상쇄 오차를 동일하게 보지만 누적 손실은 %.1f배 구분)' % (d_cum / a_cum))
