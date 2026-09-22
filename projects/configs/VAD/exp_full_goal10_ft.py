"""S2 (9/22, 메인, 마지막): 트렁크를 풀고 5초 목표점 앵커로 soup 에서 1 epoch 미세조정.

S1/S1b(동결 트렁크 + 앵커)에서 앵커 손실이 0.055 에서 전혀 내려가지 않았다 = 동결된 BEV 특징으로는
5초 목표점을 그 정밀도로 읽을 수 없다. 원인이 동결이라면 BEV 인코더·헤드를 풀어야 인코더가 목표점을
정밀하게 나르도록 배운다. 백본은 동결(계산·메모리), 나머지는 lr 0.1배(2e-5), VO 헤드 1배.
1 epoch 코사인(메인 11h). 목표점 입력 경로는 불변(BEV 조건화 한 곳), 5초 위치는 손실 라벨로만.
평가는 exp_full_tgtbev_goal10 (VO 끔). 관문: val ≤ 0.288.
"""
_base_ = ['./exp_stage2_vo_goal10_drop.py']

train_only = None
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.1),
            'pts_bbox_head': dict(lr_mult=0.1),
            'pts_bbox_head.vo_': dict(lr_mult=1.0),
        }))
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)
