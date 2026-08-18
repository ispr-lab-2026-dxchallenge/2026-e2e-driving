import torch, time
print("device:", torch.cuda.get_device_name(0))
p = torch.cuda.get_device_properties(0)
print("SM=%d  mem=%.1f GB  cc=sm_%d%d" % (p.multi_processor_count, p.total_memory/1e9, p.major, p.minor))

def bench(fn, warm=5, it=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / it

N = 4096
for name, dt, tf32 in [("fp32 (TF32 off)", torch.float32, False),
                       ("fp32 (TF32 on)",  torch.float32, True),
                       ("fp16",            torch.float16, True)]:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    a = torch.randn(N, N, device="cuda", dtype=dt)
    b = torch.randn(N, N, device="cuda", dtype=dt)
    t = bench(lambda: torch.mm(a, b))
    print("  matmul %-16s : %8.1f GFLOPS" % (name, 2*N**3/t/1e9))

conv = torch.nn.Conv2d(256, 256, 3, padding=1).cuda()
x = torch.randn(6, 256, 112, 192, device="cuda")
torch.backends.cudnn.benchmark = True
t = bench(lambda: conv(x))
print("  conv3x3 256ch 6x112x192 : %8.1f GFLOPS" % (2*6*256*256*9*112*192/t/1e9))

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction as F
v = torch.rand(1, 40000, 8, 32, device="cuda")
s = torch.tensor([[200, 200]], dtype=torch.long, device="cuda")
l = torch.tensor([0], dtype=torch.long, device="cuda")
loc = torch.rand(1, 10000, 8, 1, 8, 2, device="cuda")
a2 = torch.rand(1, 10000, 8, 1, 8, device="cuda"); a2 = a2 / a2.sum(-1, keepdim=True)
t = bench(lambda: F.apply(v, s, l, loc, a2, 64), warm=3, it=20)
print("  ms_deform_attn (10k q)  : %8.3f ms   <- PTX JIT 대상 커널" % (t*1e3))
