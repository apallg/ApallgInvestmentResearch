from .base import BaseBroker
from .fake_broker import FakeBroker
from .qmt_broker import QmtBroker
from .easytrader_broker import EasytraderBroker
from core.shared.config import BROKER_FAKE, BROKER_QMT, BROKER_EASYT


def create_broker(broker_type=BROKER_FAKE, **kwargs):
    if broker_type == BROKER_FAKE:
        return FakeBroker(**kwargs)
    if broker_type == BROKER_QMT:
        return QmtBroker(**kwargs)
    if broker_type == BROKER_EASYT:
        return EasytraderBroker(**kwargs)
    raise ValueError(f"不支持的经纪商类型: {broker_type}")


def is_fake_broker(broker):
    return isinstance(broker, FakeBroker)
