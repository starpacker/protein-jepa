#!/bin/bash
while true; do
  n=$(ps aux | grep 'cli train' | grep -v grep | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "$(date): ALL cli train finished" >> nohup_wait_stageb3.log
    for f in k4 k1; do
      echo "$f step: $(grep -o '"step": [0-9]*' nohup_stageb_v3_${f}.log | tail -1)" >> nohup_wait_stageb3.log
      echo "$f skill: $(grep -o '"paired_skill": -\?[0-9.e+-]*' nohup_stageb_v3_${f}.log | tail -1)" >> nohup_wait_stageb3.log
    done
    exit 0
  fi
  sleep 60
done
