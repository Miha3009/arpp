from experiment import Experiment
from database import PatchDataset
from models import DummyModel, LinearModel
from metrics import WeightedMSE
from optimizer import AdamOptimizer

ds = PatchDataset(
    input_variables=['inm_t2m'],
    target_variables=['t2m', 'lat'],
    modes={
        'train': {"t_min": '19910101', 't_max': '20201231', 'epoch_size': 100, 'batch_size': 8},
        'test': {"t_min": '20240901', 't_max': '20251231', 'epoch_size': 50, 'batch_size': 8},
    },
    era_scales=[
    ],
    inm_scales=[
        {'id': 'regional', 'xSize': 16, 'ySize': 16, 'tSize': 7, 'xyStep': 1, 'tStep': 1},
    ],
    target_scale={'xSize': 16, 'ySize': 16, 'tSize': 7, 'xyStep': 4, 'tStep': 1},
    fix_seed=0
)
model = LinearModel('inm_t2m_regional')
metric = WeightedMSE('t2m')
optimizer = AdamOptimizer()
experiment = Experiment('test', ds, model, metric, optimizer)
experiment.run(100)
