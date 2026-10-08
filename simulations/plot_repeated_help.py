"""Population summary and explicitly post-hoc illustrative local failure."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import experiment_repeated_help as model
from report_repeated_help import read


def main(out):
    rows=[r for r in read(out/'summary.csv') if r['n']=='96' and r['scenario']=='mixed'
          and r['sample']=='main' and r['subset']=='all']
    indexed={r['policy']:r for r in rows}
    labels=['Pause','Only if deficit persists','Pulse every 8','Pulse every 16']
    colors=['#7f8a96','#217a93','#b67b48','#80a9ad']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for ax,metric,scale,title in zip(axes,['requester_service','restricted_recovery_time','messages'],
            [100,1,1],['Requester demand served (%)','Restricted recovery time','Additional messages']):
        vals=[float(indexed[p][metric])*scale for p in model.POLICIES]
        ax.bar(range(4),vals,color=colors)
        for i,v in enumerate(vals): ax.text(i,v+max(vals)*.025,f'{v:.1f}',ha='center',fontsize=10)
        ax.set_xticks(range(4),labels,rotation=25,ha='right',fontsize=9)
        ax.set_title(title,fontsize=12)
        ax.set_ylim(0,max(vals)*1.16)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('After one successful swap: repeated help did not generally delay recovery',fontsize=14)
    fig.text(.02,.02,'96 nodes, mixed disturbances; 48 selected states / 19 eligible seeds. Equal seed weights.\n'
             '48 time units; unrecovered states assigned 48 for the restricted-time summary. Messages exclude the common first attempt.',fontsize=9)
    fig.tight_layout(rect=(0,.14,1,.93))
    fig.savefig(out/'repeated_help_comparison.png',dpi=160)
    plt.close(fig)
    with np.load(out/'example_step_traces.npz') as data:
        a=data['dt0.2_pause'][:,0]
        b=data['dt0.2_pulse8'][:,0]
    fig,ax=plt.subplots(figsize=(10,4.6))
    x=np.arange(len(a))*.2
    ax.plot(x,100*a,label='Pause after first swap (pulse16 overlaps)',color='#217a93',lw=2)
    ax.plot(x,100*b,label='Pulse8 (deficit-driven retries identical here)',color='#b67b48',lw=2)
    ax.axhline(95,color='gray',ls='--',lw=1,label='Recovery threshold: 95%')
    ax.axvline(8,color='#b67b48',ls=':',label='Additional successful swap at +8')
    ax.set(xlabel='Time since the common first swap',ylabel='Requester demand served (%)',ylim=(0,105),xlim=(0,48))
    ax.set_title('One local counterexample: network gain can coexist with requester harm')
    ax.legend(loc='lower right',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.02,.02,'Selected after inspecting results: n96 / transient / seed 24000 / t40 / node31. One recovery loss among 58 states in this group.\n'
             'The same local failure persists at dt=0.1 and 0.05. This example does not estimate its frequency outside the tested sample.',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,1))
    fig.savefig(out/'local_counterexample.png',dpi=160)
    plt.close(fig)
    (out/'plot_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=model.OUT)
    main(p.parse_args().out)
