"""인접 epoch 체크포인트 가중치 평균 (model soup / SWA 류).

같은 실행의 epoch_N.pth 여러 개를 단순 평균해 새 .pth 로 저장한다. 구조·입력·FLOPs·T_infer 는
단일 모델과 동일. 백본은 norm_eval=True 라 BN 통계가 전 체크포인트에서 같으므로 재추정 불필요.
정수/bool 텐서(num_batches_tracked 등)는 첫 체크포인트 값을 그대로 쓴다.

  python tools/avg_ckpt.py --out work_dirs/x/epoch_21.pth work_dirs/x/epoch_5.pth work_dirs/x/epoch_6.pth work_dirs/x/epoch_7.pth
"""
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    ckpts = [torch.load(f, map_location='cpu', weights_only=False) for f in a.inputs]
    sds = [c['state_dict'] for c in ckpts]
    keys = list(sds[0].keys())
    for sd in sds[1:]:
        assert list(sd.keys()) == keys, '체크포인트 키가 다르다'
    avg = {}
    n_avg = 0
    for k in keys:
        t0 = sds[0][k]
        if t0.is_floating_point():
            avg[k] = sum(sd[k].float() for sd in sds).div_(len(sds)).to(t0.dtype)
            n_avg += 1
        else:
            avg[k] = t0.clone()
    meta = dict(ckpts[0].get('meta', {}))
    meta['soup_of'] = list(a.inputs)
    torch.save({'state_dict': avg, 'meta': meta}, a.out)
    print('평균 텐서 %d / 전체 %d  ->  %s  (원본: %s)' % (n_avg, len(keys), a.out, ', '.join(a.inputs)))


if __name__ == '__main__':
    main()
