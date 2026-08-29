import os
import os.path as osp
import platform
import shutil

import mmcv
from mmcv.runner import save_checkpoint
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from projects.mmdet3d_plugin.models.utils import run_time
from mmcv.parallel import is_module_wrapper


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())

@HOOKS.register_module()
class CustomSetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch)


@HOOKS.register_module()
class IterSafeCheckpointHook(CheckpointHook):
    """epoch 중간에 저장해도 재개 시 그 epoch 을 건너뛰지 않는 체크포인트 훅.

    왜 필요한가
    -----------
    mmcv 의 ``EpochBasedRunner.save_checkpoint`` 는 메타에 ``epoch + 1`` 을 적는다.
    epoch 이 실제로 끝났을 때를 가정한 값이다. 그런데 ``by_epoch=False`` 로
    iteration 마다 저장하면 **epoch 중간에도 "이 epoch 은 끝났다"고 기록**된다.

    거기서 재개하면 runner 는 그 epoch 이 완료된 줄 알고 다음 epoch 으로 넘어간다.
    2026-08-19 실측으로 ``Epoch [1][3700/20220]`` 진행 중이던 학습이 재개 후
    ``Epoch [2][100/20220]`` 이 됐고, 코사인 LR 도 한 칸 진행됐다.
    즉 **재개 1회당 epoch 을 하나씩 까먹는다.** 그날 하루에만 3번 일어났다.

    무엇이 달라지나
    ---------------
    - 메타에 ``epoch`` 을 그대로 적는다 -> 재개하면 그 epoch 을 **처음부터 다시** 돈다.
      이미 계산한 만큼은 다시 하지만, **epoch 회계와 LR 스케줄이 정확해진다.**
    - 파일명을 ``iter_{iter}.pth`` 로 짓는다. 기존 ``epoch_{n}.pth`` 는 한 epoch 안에서
      같은 이름을 계속 덮어써서 ``max_keep_ckpts`` 가 무의미했고, 저장 도중 죽으면
      **유일한 복구점이 깨질 위험**이 있었다. 이제 3개가 번갈아 남는다.

    사용법::

        checkpoint_config = dict(
            type='IterSafeCheckpointHook',
            by_epoch=False, interval=1000, max_keep_ckpts=3)
    """

    def after_train_epoch(self, runner):
        if self.by_epoch:
            return super().after_train_epoch(runner)
        # epoch 경계 보존: 학습량 곡선(과적합 판별) 평가용 (RESULTS.md #10).
        # iter_*.pth 프루닝과 무관하게 남고, optimizer 를 빼 용량을 줄인다.
        # 재개는 계속 latest.pth(iter 파일)를 쓴다.
        meta = dict(epoch=runner.epoch + 1, iter=runner.iter)
        if runner.meta is not None:
            meta.update(runner.meta)
        out_dir = self.out_dir or runner.work_dir
        filepath = osp.join(out_dir, 'epoch_{}.pth'.format(runner.epoch + 1))
        save_checkpoint(runner.model, filepath, optimizer=None, meta=meta)

    def _save_checkpoint(self, runner):
        if self.by_epoch:
            # epoch 경계 저장은 mmcv 기본 동작이 이미 옳다
            return super()._save_checkpoint(runner)

        meta = dict(epoch=runner.epoch, iter=runner.iter)
        if runner.meta is not None:
            meta.update(runner.meta)

        out_dir = self.out_dir or runner.work_dir
        filename = 'iter_{}.pth'.format(runner.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = runner.optimizer if self.save_optimizer else None
        save_checkpoint(runner.model, filepath, optimizer=optimizer, meta=meta)

        # latest.pth 심볼릭 링크 (train_watch.sh 와 --resume-from 이 이걸 본다)
        dst = osp.join(out_dir, 'latest.pth')
        if platform.system() != 'Windows':
            mmcv.symlink(filename, dst)
        else:
            shutil.copy(filepath, dst)

        runner.meta = runner.meta or {}
        runner.meta.setdefault('hook_msgs', {})
        runner.meta['hook_msgs']['last_ckpt'] = filepath

        # 오래된 것 정리
        if self.max_keep_ckpts > 0:
            redundant = runner.iter + 1 - (self.max_keep_ckpts * self.interval)
            for i in range(redundant, 0, -self.interval):
                old = osp.join(out_dir, 'iter_{}.pth'.format(i))
                if osp.exists(old):
                    os.remove(old)
                else:
                    break
