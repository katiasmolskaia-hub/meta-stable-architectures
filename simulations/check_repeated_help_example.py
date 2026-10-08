"""Post-hoc step check of the observed local recovery loss; not a new primary test."""
import argparse
from pathlib import Path
import numpy as np
import experiment_repeated_help as model
import experiment_local_support_signal as base


def main(out):
    if (out/'example_step.csv').exists():
        raise FileExistsError(out/'example_step.csv')
    rows,traces=[],{}
    for dt in (.2,.1,.05):
        case=(96,'transient',24000,dt)
        state,p,supply,demand=model.reconstruct(case,40.,31)
        for policy in model.POLICIES:
            r,ev,tr=model.branch(state,'transient',24000,40.,31,policy,p,supply,demand)
            rows.append(dict(n=96,scenario='transient',seed=24000,dt=dt,t=40.,node=31,**r))
            traces[f'dt{dt}_{policy}']=tr
    base.write_csv(out/'example_step.csv',rows)
    np.savez_compressed(out/'example_step_traces.npz',**traces)
    (out/'example_check_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for r in rows:
        print({k:r[k] for k in ('dt','policy','requester_service','recovered','restricted_recovery_time','served_units','swaps')})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=model.OUT)
    main(p.parse_args().out)
