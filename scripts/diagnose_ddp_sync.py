"""Reproduce the trainer's DDP bypass using two CPU workers (no training files)."""
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def worker(rank, port):
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2)
    for bypass in (True, False):
        model = DDP(torch.nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model.module.weight.fill_(1.)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for step in range(2):
            optimizer.zero_grad()
            x = torch.tensor([[float(rank + 1)]])
            output = model.module(x) if bypass else model(x)
            output.sum().backward()
            grads = [torch.zeros_like(model.module.weight) for _ in range(2)]
            dist.all_gather(grads, model.module.weight.grad)
            optimizer.step()
            weights = [torch.zeros_like(model.module.weight) for _ in range(2)]
            dist.all_gather(weights, model.module.weight.detach())
            expected = [1., 2.] if bypass else [1.5, 1.5]
            assert [g.item() for g in grads] == expected
            if rank == 0:
                print(dict(bypass=bypass, step=step, grads=[g.item() for g in grads],
                           weights=[w.item() for w in weights]), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', type=int, default=29687)
    args = parser.parse_args()
    mp.start_processes(worker, args=(args.port,), nprocs=2, start_method='fork')
