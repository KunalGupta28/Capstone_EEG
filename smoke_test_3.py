import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\Dell\Desktop\EEG')
os.chdir(r'c:\Users\Dell\Desktop\EEG')

from models.models  import MODEL_REGISTRY
from pipeline.utils import set_seed, get_device
from pipeline.train import train_and_evaluate_subject
from pipeline.eval import print_metrics_table

CONFIG = {
    'n_folds': 2, 'epochs': 1, 'batch_size': 32,
    'lr': 1e-3, 'weight_decay': 1e-4,
    'early_stopping_patience': 5, 'lr_scheduler_patience': 2,
    'seed': 42, 'dropout': 0.5,
}

set_seed(42)
device = get_device()

print("="*60)
print("  Running Demo for BCI_III_IVa (Binary Dataset)")
print("="*60)

all_results = {}

subj_binary = 'BCICIV_ds3_aa'
print(f"\n---> Starting BCI_III_IVa ({subj_binary})")
results_1 = train_and_evaluate_subject(
    subject_id     = subj_binary,
    processed_dir  = r'c:\Users\Dell\Desktop\EEG\processed_data\BCI-III-IVa-preprocessed',
    results_dir    = r'c:\Users\Dell\Desktop\EEG\results',
    config         = CONFIG,
    model_registry = {"EEGNet": MODEL_REGISTRY["EEGNet"], "LSTM": MODEL_REGISTRY["LSTM"]},
    device         = device,
)
all_results[subj_binary] = results_1

print("\n=== FINAL RESULTS ===")
print_metrics_table(all_results)
