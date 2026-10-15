#!/bin/bash
cd /liuyutian/protein_jepa_frame
./launch_opf_stage_a.sh > nohup_opf_stage_a_master.log 2>&1 &
echo "stage-A launched pid $!"
