from yacs.config import CfgNode

# ---------------------------------------------------------
# Config definition
# ---------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.ROOT = "raw_data"
_C.DATASET.NAME = "LGG"  # Options: "BLCA", "LGG", "RCC"
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.CLASS_NAMES = ["Grade II", "Grade III"]
_C.DATASET.NUM_FOLDS = 10  # For k-fold cross-validation
_C.DATASET.OMICS = ["DNA", "mRNA", "miRNA"]
_C.DATASET.FEATURE_SPARSITY_RATES = [0.9, 0.9, 0.8]  # For each modality: DNA, mRNA, miRNA, respectively
_C.DATASET.PATIENT_SPARSITY_RATES = 0.85  # For constructing the patient similarity network

# ---------------------------------------------------------
# Feature selection module using ACO
# ---------------------------------------------------------
_C.ACO = CfgNode()
_C.ACO.MAX_ITERS = 50
_C.ACO.NUM_AGENTS = 10
_C.ACO.INIT_PHEROMONE = 0.2
_C.ACO.Q0 = 0.8
_C.ACO.NODE_DISC_RATE = 0.1
_C.ACO.EDGE_DISC_RATE = 0.1
_C.ACO.PROB_DISC_RATE = 0.1
_C.ACO.FIX_FEAT_SIZE = 30  # Number of features selected per agent per iteration
_C.ACO.FINAL_FEAT_SIZES = [60, 105, 165, 300]  # Sizes of final selected feature subsets
_C.ACO.SELECTION_RATE = 0.8  # Pheromone and relevance contribution to final feature subset selection

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.MAX_EPOCHS_PRETRAIN = 500
_C.SOLVER.MAX_EPOCHS = 500
_C.SOLVER.SEED = 24
_C.SOLVER.TUNE_HYPER = False

# ---------------------------------------------------------
# GAT module
# ---------------------------------------------------------
_C.GAT = CfgNode()
_C.GAT.NUM_LAYERS = 3
_C.GAT.NUM_HEADS = 2
_C.GAT.HIDDEN_DIM = [100, 100, 50]
_C.GAT.DROPOUT_RATE = 0.3
_C.GAT.LR_PRETRAIN = 1e-3
_C.GAT.LR = 1e-3  # Learning rate
_C.GAT.WD = 1e-3  # Weight decay
_C.GAT.FINAL_FEAT_SIZES = [300]

# ---------------------------------------------------------
# VCDN module
# ---------------------------------------------------------
_C.VCDN = CfgNode()
_C.VCDN.LR = 5e-2  # Learning rate
_C.VCDN.WD = 1e-3  # Weight decay

# ---------------------------------------------------------
# Biomarker identification
# ---------------------------------------------------------
_C.BIOMK = CfgNode()
_C.BIOMK.FEAT_SIZE = 300
_C.BIOMK.NUM_TOP_BIOMARKERS = 30

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
_C.RESULT = CfgNode()
_C.RESULT.OUTPUT_DIR = "output"
# setup GAT model directory
_C.RESULT.SAVE_MODEL_DIR = "model"
_C.RESULT.SAVE_MODEL_TMPL = "{dataset_name}_model_[{fold_idx}]_[{feat_size}].ckpt"
_C.RESULT.SAVE_MODEL = False
# setup rich data directory
_C.RESULT.SAVE_RICH_DATA_DIR = "rich_data"
_C.RESULT.SAVE_RICH_DATA_TMPL = "{dataset_name}_data_[{fold_idx}].pkl"
_C.RESULT.SAVE_RICH_DATA = False
# setup output csv files
_C.RESULT.FILE_HEADER_CLF = ['feature size', 'classifier', 'metric', 'fold', 'value']
_C.RESULT.FILE_HEADER_TIME = ['feature size', 'fold', 'value']
_C.RESULT.FILE_HEADER_GAT = ['feature size', 'metric', 'fold', 'value']
_C.RESULT.FILE_HEADER_GAT_TIME = ['feature size', 'fold', 'value']
_C.RESULT.SAVE_RESULT = True
# setup PyTorch Lightning log
_C.RESULT.LIGHTNING_LOG_DIR = "lightning"


def get_cfg_defaults():
    return _C.clone()
