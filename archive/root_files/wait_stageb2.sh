#!/bin/bash
while true; do
  n=$(ps aux | grep 'cli train' | grep -v grep | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "$(date): ALL cli train processes finished" >> nohup_wait_stageb2.log
    # capture final skill lines
    for f in k4 k1; do
      echo "$f final: $(grep -o '"paired_skill": [0-9.e+-]*' nohup_opf_${f}_stageb2.log | tail -1)" >> nohup_wait_stageb2.log
      echo "$f step:  $(grep -o '"step": [0-9]*' nohup_opf_${f}_stageb2.log | tail -1)" >> nohup_wait_stageb2.log
      echo "$f orth:  $(grep -o '"opf_orth_loss": [0-9.e+-]*' nohup_opf_${f}_stageb2.log | tail -1)" >> nohup_wait_stageb2.log
    done
    exit 0
  fi
  echo "$(date): still running, $n procs" >> nohup_wait_stageb2.log
  sleep 60
done
