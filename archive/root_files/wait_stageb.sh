#!/bin/bash
cd /liuyutian/protein_jepa_frame
for i in $(seq 1 200); do
  n=$(ps aux | grep "cli train" | grep -v grep | wc -l)
  if [ "$n" -eq "0" ]; then echo "STAGE-B DONE"; break; fi
  sleep 20
done
echo "STAGE-B DONE at poll $i, procs=$(ps aux | grep 'cli train' | grep -v grep | wc -l)"
