"""Offline sequence-grouped input selection; never used by online inference."""
from statistics import mean
from .lidar_inference_config import RawLidarConfig


def select_grouped_inputs(records, precision_floor=.96, recall_floor=.85):
    """Select one setting using equal sequence weight and report leave-one-out.

    Failing the joint precision/recall guard produces an experimental fallback.
    Rank its worst sequence F1 first, then the mean: a long/easy sequence cannot
    compensate for a failed short sequence. The held-out fold never selects its
    own input setting. This measures sensitivity, not statistical certification.
    """
    groups={}
    for row in records:
        sequence=row['sequence']
        setting=(row['scale'],row['sweeps'])
        RawLidarConfig(intensity_scale=setting[0],sweeps=setting[1])
        key=(sequence,setting)
        if key in groups:raise ValueError('duplicate sequence/input setting')
        score=row['observed']
        for field in ('precision','recall','f1'):
            value=score[field]
            if value is not None and not 0<=value<=1:raise ValueError('invalid development score')
        groups[key]=row
    sequences=sorted({s for s,_ in groups})
    settings=sorted({p for _,p in groups})
    if len(sequences)<3 or len(settings)<2:
        raise ValueError('at least three development sequences and two input settings required')
    if any((s,p) not in groups for s in sequences for p in settings):
        raise ValueError('incomplete sequence/input grid')

    def choose(training):
        eligible=[p for p in settings if all(
            (groups[s,p]['observed']['precision'] or 0)>=precision_floor and
            (groups[s,p]['observed']['recall'] or 0)>=recall_floor for s in training)]
        def rank(p):
            field='recall' if eligible else 'f1'
            values=[groups[s,p]['observed'][field] or 0 for s in training]
            # Deterministic tie-break only; prefer fewer sweeps, then lower scale.
            return min(values),mean(values),-p[1],-p[0]
        return max(eligible or settings,key=rank),bool(eligible)

    selected,passed=choose(sequences)
    folds=[]
    for held_out in sequences:
        training=[s for s in sequences if s!=held_out]
        setting,fold_passed=choose(training)
        folds.append(dict(training_sequences=training,held_out_sequence=held_out,
                          selected_scale=setting[0],selected_sweeps=setting[1],
                          training_acceptance_passed=fold_passed,
                          held_out_observed=groups[held_out,setting]['observed']))
    return dict(development_sequences=sequences,
                selected_inference_config=dict(intensity_scale=selected[0],sweeps=selected[1]),
                development_acceptance_passed=passed,folds=folds,
                fold_selection_agrees_with_final=all((f['selected_scale'],f['selected_sweeps'])==selected for f in folds),
                selection='joint P/R guard on every training sequence; else maximize worst sequence F1, then equally weighted mean F1')
