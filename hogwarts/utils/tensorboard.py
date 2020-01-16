__all__ = ['init', 'checkpoint', 'truncate']

import struct
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.event_pb2 import Event


def init(log_dir, trunc_anchor=0):
    if trunc_anchor > 0:
        truncate(log_dir, trunc_anchor)
    writer = SummaryWriter(log_dir)
    return writer


def checkpoint(writer, anchor):
    writer.add_scalar('hogwarts/progress', anchor, anchor)


def truncate(log_dir, trunc_anchor):
    log_dir = Path(log_dir)
    for event_fpath in log_dir.glob('events*'):
        with event_fpath.open('rb+') as f:
            trunc_length = 0
            while True:
                length = f.read(8)
                if len(length) == 0: break
                length = struct.unpack('Q', length)[0]
                f.read(4)
                event = f.read(length)
                event = Event.FromString(event)
                f.read(4)
                trunc_length += 16 + length
                try:
                    tag = event.summary.value[0].tag
                    value = event.summary.value[0].simple_value
                    if tag == 'hogwarts/progress' and value == trunc_anchor:
                        break
                except Exception:
                    pass

            f.seek(0)
            f.truncate(trunc_length)
