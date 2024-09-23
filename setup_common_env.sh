######################################################################
# Script to setup the environment variables
# Usage: source setup_personal_env.sh && source setup_common_env.sh
######################################################################

# environment variables
export TINYLM_WORK_DIR=${PERSONAL_WORKSPACE_DIR}/prox
export EVAL_ANALYSIS_DIR=${TINYLM_WORK_DIR}/eval_analysis
export RAW_DATA_DIR=${PERSONAL_STORAGE_DIR}/prox/raw_data
export TOKENIZE_DATA_DIR=${PERSONAL_STORAGE_DIR}/prox/tokenize_data
export ADAPT_MODEL_OUTPUT_DIR=${PERSONAL_STORAGE_DIR}/prox/adapt
export PT_MODEL_OUTPUT_DIR=${PERSONAL_STORAGE_DIR}/prox/pt
export HF_MODEL_OUTPUT_DIR=${PERSONAL_STORAGE_DIR}/prox/hf
export DATA_ROOT_DIR=${PERSONAL_STORAGE_DIR}
mkdir -p $TINYLM_WORK_DIR
mkdir -p $EVAL_ANALYSIS_DIR
mkdir -p $RAW_DATA_DIR
mkdir -p $TOKENIZE_DATA_DIR
mkdir -p $ADAPT_MODEL_OUTPUT_DIR
mkdir -p $PT_MODEL_OUTPUT_DIR
mkdir -p $HF_MODEL_OUTPUT_DIR
mkdir -p $DATA_ROOT_DIR

# seed
export SEED=42

# alias for enter dir
alias cd_tinylm='cd $TINYLM_WORK_DIR'
alias cd_rawdata='cd $RAW_DATA_DIR'
alias cd_tokenize='cd $TOKENIZE_DATA_DIR'
alias cd_ptmodel='cd $PT_MODEL_OUTPUT_DIR'
alias cd_hfmodel='cd $HF_MODEL_OUTPUT_DIR'

# activate PT environment
# define alias for 2 commands:
alias act_pt='source ${TINYLM_CONDA_DIR} && conda activate ${TINYLM_ENV_NAME}'