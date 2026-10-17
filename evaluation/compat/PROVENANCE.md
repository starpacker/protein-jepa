# Vendored legacy code provenance

Both files were copied from git `HEAD` = `07d07a6` (`refactor(stage-ab): replace
old ABC pipeline with Stage A v2 + B1a redesign`):

    git show HEAD:protein_jepa_frame/models/stage_c/structure_decoder.py \
        > evaluation/compat/structure_decoder_v1.py
    git show HEAD:protein_jepa_frame/models/stage_b1a/model.py \
        > evaluation/compat/stage_b1a_v1.py

Only edit applied: `stage_b1a_v1.py` imports `DecoderConfig`/`StructureDecoder`
from `.structure_decoder_v1` instead of from the (now-rewritten) working-tree
`protein_jepa_frame.models.stage_c.structure_decoder`.

Why: the working tree carries an uncommitted rewrite of `StructureDecoder` that
renames/removes the heads (`h_input_norm`/`q_input_norm`/`q_position_head`/...)
present in every B1a oracle checkpoint. Without these vendored copies, those
checkpoints cannot be loaded at all.

Selection is automatic: `evaluation/loaders.py` uses v1 iff the checkpoint's
oracle state dict contains `decoder.h_input_norm.weight`.
