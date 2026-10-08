"""Audit repeated-help branches and report paired, seed-clustered contrasts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import experiment_repeated_help as model
import experiment_local_support_signal as base
import experiment_minimal_observer as prior
import experiment_help_timing as timing


def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def key(r):
    return (int(r['n']),r['scenario'],int(r['seed']),float(r['dt']),float(r['t']),int(r['node']))


def group(r):
    return (int(r['n']),r['scenario'],float(r['dt']),'main' if int(r['seed'])<24100 else 'step')


def interval(values, rng):
    v = np.asarray(values)
    means = v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1)
    return float(v.mean()), *map(float,np.quantile(means,[.025,.975]))


def report(out):
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    for name,h in config['hashes'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h,name
    for module in (model,base,prior,timing):
        p=Path(module.__file__)
        assert p.read_bytes().replace(b'\r\n',b'\n')==(out/p.name).read_bytes().replace(b'\r\n',b'\n')
    rows,sel,events=(read(out/name) for name in ('runs.csv','selections.csv','events.csv'))
    expected={tuple(c)+(t,) for c in config['cases'] for t in model.TIMES}
    assert len(sel)==len(expected) and {key(r)[:-1] for r in sel}==expected
    selected={key(r) for r in sel if int(r['first_swap'])}
    indexed={(key(r),r['policy']):r for r in rows}
    assert len(rows)==len(indexed)==len(selected)*4
    assert set(indexed)=={(k,p) for k in selected for p in model.POLICIES}
    evmap={k:[] for k in indexed}
    for e in events:
        evmap[key(e),e['policy']].append(e)
    trace_count=0
    for case in config['cases']:
        n,scenario,seed,dt=case
        matching=[r for r in rows if key(r)[:4]==tuple(case)]
        with np.load(out/f'traces_{n}_{scenario}_{seed}_{dt}.npz') as traces:
            assert set(traces.files)=={f"t{int(float(r['t']))}_{r['policy']}" for r in matching}
            for r in matching:
                tr=traces[f"t{int(float(r['t']))}_{r['policy']}"]
                assert tr.shape==(round(48/dt),4)
                assert abs(tr[:,1].sum()-float(r['served_units']))<1e-9
                recovered,tail=model.recovery(tr[:,0],dt)
                assert recovered==int(r['recovered']) and tail==float(r['restricted_recovery_time'])
                if r['policy']!='pause':
                    pause=traces[f"t{int(float(r['t']))}_pause"]
                    np.testing.assert_array_equal(tr[:round(8/dt)],pause[:round(8/dt)])
                trace_count+=1
    for r in sel:
        assert int(r['first_messages'])==(0 if int(r['node'])<0 else 24+8*int(r['first_swap']))
        assert abs(float(r['first_fee'])-.12*int(r['first_swap']))<1e-12
    for r in rows:
        ev=evmap[key(r),r['policy']]
        assert len(ev)==int(r['attempts'])<=5
        assert sum(int(e['swap']) for e in ev)==int(r['swaps'])
        assert int(r['messages'])==24*len(ev)+8*int(r['swaps'])
        assert abs(float(r['fee'])-.12*int(r['swaps']))<1e-12
        assert float(r['max_mass_error'])<1e-8 and float(r['max_flow_ratio'])<=1+1e-9
        times=[0.]+[float(e['delay']) for e in ev]
        assert all(b-a>=8-1e-9 for a,b in zip(times,times[1:]))
        if r['policy']=='persistent': assert all(int(e['need']) for e in ev)
        if r['policy']=='pause': assert not ev
        if r['policy'].startswith('pulse'):
            assert times[1:]==list(range(int(r['policy'][5:]),48,int(r['policy'][5:])))
    replay=0
    for g in sorted({group(r) for r in rows}):
        first=next(r for r in rows if group(r)==g)
        n,scenario,seed,dt,t,node=key(first)
        state,p,supply,demand=model.reconstruct((n,scenario,seed,dt),t,node)
        with np.load(out/f'traces_{n}_{scenario}_{seed}_{dt}.npz') as traces:
            for policy in model.POLICIES:
                result,ev,tr=model.branch(state,scenario,seed,t,node,policy,p,supply,demand)
                saved=indexed[key(first),policy]
                assert all(saved[k]==str(v) for k,v in result.items())
                assert len(ev)==len(evmap[key(first),policy])
                for actual,stored in zip(ev,evmap[key(first),policy]):
                    assert all(stored[k]==str(v) for k,v in actual.items())
                np.testing.assert_array_equal(tr,traces[f't{int(t)}_{policy}'])
                replay+=1
    metrics=('served_units','service','requester_service','requester_deficit','other_served_units',
             'recovered','restricted_recovery_time','full_final_recovery','attempts','swaps','messages','fee')
    summary,contrasts,deltas=[],[],[]
    rng=np.random.default_rng(24501)
    for k in sorted(selected):
        pause=indexed[k,'pause']
        for policy in model.POLICIES[1:]:
            r=indexed[k,policy]
            fields={f:r[f] for f in ('n','scenario','seed','dt','t','node')}
            ds={m:float(r[m])-float(pause[m]) for m in metrics}
            deltas.append(dict(**fields,policy=policy,**ds,
                harmful=int(ds['served_units']<-.01),helpful=int(ds['served_units']>.01),
                tail_longer4=int(ds['restricted_recovery_time']>=4-1e-9),
                lost_recovery=int(int(pause['recovered'])==1 and int(r['recovered'])==0),
                gained_recovery=int(int(pause['recovered'])==0 and int(r['recovered'])==1)))
    for g in sorted({group(r) for r in rows}):
        for subset in ('all','post100'):
            subset_rows=[r for r in rows if group(r)==g and (subset=='all' or float(r['t'])>=100)]
            seeds=sorted({int(r['seed']) for r in subset_rows})
            if not seeds: continue
            fields=dict(n=g[0],scenario=g[1],dt=g[2],sample=g[3],subset=subset,seeds=len(seeds),
                        states=len(subset_rows)//4)
            for policy in model.POLICIES:
                selected_rows=[r for r in subset_rows if r['policy']==policy]
                means={m:float(np.mean([np.mean([float(r[m]) for r in selected_rows if int(r['seed'])==s])
                        for s in seeds])) for m in metrics}
                summary.append(dict(**fields,policy=policy,**means))
            for a,b in (('pulse8','pause'),('persistent','pause'),('pulse16','pause'),('pulse8','persistent')):
                for m in metrics:
                    values=[]
                    for s in seeds:
                        states=[key(r) for r in subset_rows if int(r['seed'])==s and r['policy']=='pause']
                        values.append(np.mean([float(indexed[k,a][m])-float(indexed[k,b][m]) for k in states]))
                    mean,lo,hi=interval(values,rng)
                    contrasts.append(dict(**fields,contrast=a+'-'+b,metric=m,mean=mean,lo=lo,hi=hi))
    base.write_csv(out/'summary.csv',summary)
    base.write_csv(out/'paired_comparisons.csv',contrasts)
    base.write_csv(out/'deltas.csv',deltas)
    audit=dict(configurations=len(config['cases']),slots=len(sel),empty_slots=sum(int(r['node'])<0 for r in sel),
        first_failed=sum(int(r['node'])>=0 and not int(r['first_swap']) for r in sel),
        selected_successes=len(selected),branches=len(rows),traces_checked=trace_count,exact_replays=replay,
        max_mass_error=max(float(r['max_mass_error']) for r in rows),hashes_verified=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    (out/'report_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(audit))
    for r in summary:
        if r['n']==96 and r['scenario']=='mixed' and r['sample']=='main' and r['subset']=='all': print(r)
    for r in contrasts:
        if r['subset']=='all' and r['contrast']=='pulse8-pause' and r['metric'] in ('served_units','restricted_recovery_time'):
            print(r)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=model.OUT)
    report(p.parse_args().out)
