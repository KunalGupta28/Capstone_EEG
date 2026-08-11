import sys, os, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from models.models  import MODEL_REGISTRY
from pipeline.utils import set_seed, get_device
from pipeline.train import train_and_evaluate_subject

CONFIG = {
    'n_folds': 2, 'epochs': 2, 'batch_size': 32,
    'lr': 1e-3, 'weight_decay': 1e-4,
    'early_stopping_patience': 5, 'lr_scheduler_patience': 2,
    'seed': 42, 'dropout': 0.5,
}

def main():
    set_seed(42)
    device = get_device()

    results = train_and_evaluate_subject(
        subject_id     = 'BCICIV_calib_ds1a',
        processed_dir  = os.path.join(ROOT, 'processed_data', 'BCI_IV_1_mat-preprocessed'),
        results_dir    = os.path.join(ROOT, 'results'),
        config         = CONFIG,
        model_registry = MODEL_REGISTRY,
        device         = device,
    )

    print()
    print('=== SMOKE TEST RESULTS ===')
    for model, m in results.items():
        acc = m['accuracy']
        f1  = m['f1']
        auc = m.get('roc_auc', 0.0)
        print(f'  {model:10s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}')
    print('SMOKE TEST PASSED')

if __name__ == "__main__":
    main()
