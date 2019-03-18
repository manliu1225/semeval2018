#! /bin/bash
FOLD=5
work_path=$(pwd)
for I in $(seq 1 ${FOLD}); do
  echo "Folder ${I} ..."
  for J in $(seq 1 ${FOLD}); do
    if [ ${I} -ne ${J} ]; then
      cat data/data-split-20171027/${J}/train_${J}.text >> ${work_path}/train_${I}.text
      cat data/data-split-20171027/${J}/train_${J}.labels >> ${work_path}/train_${I}.labels
    fi
  done
  cat data/data-split-20171027/${I}/train_${I}.text > ${work_path}/evaluation_${I}.text
  cat data/data-split-20171027/${I}/train_${I}.labels > ${work_path}/evaluation_${I}.labels
  ls
  python test.py --X_train ${work_path}/train_${I}.text  --y_train ${work_path}/train_${I}.labels  --X_test ${work_path}/evaluation_${I}.text --y_test ${work_path}/evaluation_${I}.labels &> bb${I}.log &
  # rm -f ${work_path}/train_${I}.text ${work_path}/train_${I}.labels ${work_path}/evaluation_${I}.text ${work_path}/evaluation_${I}.labels
done



