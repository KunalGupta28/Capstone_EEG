import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\Dell\Desktop\EEG')
os.chdir(r'c:\Users\Dell\Desktop\EEG')

from models.models  import MODEL_REGISTRY
from pipeline.utils import set_seed, get_device
from pipeline.train import train_and_evaluate_subject

CONFIG = {
    'n_folds': 2, 'epochs': 1, 'batch_size': 32,
    'lr': 1e-3, 'weight_decay': 1e-4,
    'early_stopping_patience': 5, 'lr_scheduler_patience': 2,
    'seed': 42, 'dropout': 0.5,
}

set_seed(42)
device = get_device()

# Just run EEGNet and CNN to test shapes and multi-class logic
smoke_registry = {
    'EEGNet': MODEL_REGISTRY['EEGNet'],
    'CNN': MODEL_REGISTRY['CNN']
}

results = train_and_evaluate_subject(
    subject_id     = 'A01T',
    processed_dir  = r'c:\Users\Dell\Desktop\EEG\processed_data',
    results_dir    = r'c:\Users\Dell\Desktop\EEG\results',
    config         = CONFIG,
    model_registry = smoke_registry,
    device         = device,
)

print()
print('=== SMOKE TEST RESULTS ===')
for model, m in results.items():
    acc = m['accuracy']
    f1  = m['f1']
    print(f'  {model:10s}  Acc={acc:.4f}  F1={f1:.4f}')
print('SMOKE TEST PASSED')
