results_path="./test"  # replace to your results path
batch_size=8
weight_path="checkpoint_best.pt"

TASK="PCBA" # DUDE or PCBA

python ./unimol/test.py --user-dir ./unimol "./data" --valid-subset test \
       --results-path "$results_path" \
       --num-workers 8 --batch-size "$batch_size" \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --seed 1 \
       --path "$weight_path" \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task "$TASK" \
       "$@"
