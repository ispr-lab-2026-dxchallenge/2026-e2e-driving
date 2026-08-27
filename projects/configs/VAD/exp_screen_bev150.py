"""스크리닝: BEV 격자 100x100 -> 150x150.

같은 인지 범위(전방 +-30m, 좌우 +-15m)를 더 촘촘한 격자로 본다.
셀 하나가 0.6x0.3m -> 0.4x0.2m 가 되어 차선·객체의 공간 해상도가 올라간다.
BEVFormer 계열에서 격자 해상도는 인식 성능에 일관되게 기여하는 것으로 보고돼 있다.
순수 비전 스케일링이라 규정 리스크가 없다.

BEV 크기는 config 여러 곳에 박혀 있어 전부 함께 바꿔야 한다:
  head.bev_h/bev_w, transformer.rotate_center, positional_encoding 의
  row/col_num_embed, dataset 의 bev_size.
하나라도 빼먹으면 크기 불일치로 죽거나(운 좋음) 조용히 어긋난다(운 나쁨).

비교 대상: exp_screen_base (1/3 데이터, 4 epoch 동일)
주의: 3060 12GB 에서는 배치를 낮춰야 할 수 있다. 배치가 다르면
     비교에 교란이 생기므로 결과 해석 시 명시할 것.
"""
_base_ = ['./exp_screen_base.py']

_bev_ = 150

model = dict(
    pts_bbox_head=dict(
        bev_h=_bev_,
        bev_w=_bev_,
        transformer=dict(
            rotate_center=[_bev_ // 2, _bev_ // 2],
        ),
        positional_encoding=dict(
            row_num_embed=_bev_,
            col_num_embed=_bev_,
        ),
    ))

data = dict(
    train=dict(bev_size=(_bev_, _bev_)),
    val=dict(bev_size=(_bev_, _bev_)),
    test=dict(bev_size=(_bev_, _bev_)),
)
