"""Regression tests for source semantics and rejection of deceptive training gains."""
import copy
import numpy as np
import pytest
from motor.benchmark import compare
from motor.dog_mocap import read_bvh, positions, is_holdout


def _bvh():
    return '''HIERARCHY
ROOT root {
 OFFSET 0 10 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT child {
  OFFSET 2 0 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site { OFFSET 1 0 0 }
 }
}
MOTION
Frames: 1
Frame Time: 0.0083333333
0 3 0 90 90 0 0 0 0
'''


def test_bvh_channel_order_and_local_translations():
    joints, frames, dt = read_bvh(_bvh())
    p = positions(joints, frames)
    # Rz(90) Rx(90) sends x to y, y to z. Position channels replace
    # the static root offset (the source Biped exporter stores local translations).
    np.testing.assert_allclose(p[0,0], [0,3,0], atol=1e-7)
    np.testing.assert_allclose(p[0,1], [0,5,0], atol=1e-7)
    np.testing.assert_allclose(p[0,2], [0,6,0], atol=1e-7)
    assert abs(dt - 1/120) < 1e-8
    # A y offset distinguishes intrinsic channel order from reversed Euler order.
    joints[1]['offset'] = np.array([0.,2.,0.])
    np.testing.assert_allclose(positions(joints,frames)[0,1], [0,3,2], atol=1e-7)


def test_bvh_rejects_truncated_or_nonfinite_channel_data():
    with pytest.raises(ValueError, match='channel data'):
        read_bvh(_bvh().replace('0 3 0 90 90 0 0 0 0','0 3'))
    with pytest.raises(ValueError, match='channel data'):
        read_bvh(_bvh().replace('0 3 0 90 90 0 0 0 0','nan 3 0 90 90 0 0 0 0'))


def test_split_is_per_whole_clip_not_frame():
    assert is_holdout('walk.bvh') == is_holdout('walk.bvh')
    flags = [is_holdout(f'walk_{i}.bvh') for i in range(100)]
    assert 5 < sum(flags) < 40


def _report():
    return dict(schema='skeleto-nn/motor-benchmark/1', family='quadruped',
                seconds=8., seeds=[101,202,303], harness=0., environment={}, score=.5,
                cases={n:dict(score=.5,survival=1.,failures=0,goal=.2,tilt_p95_deg=10.,facing_mean=.9,slip_p95_m_s=.2,contact_fraction=1.,flight_height_error_spans=None)
                       for n in ('dog_walk','horse_walk')})


def test_mean_improvement_cannot_hide_lost_species():
    old=_report(); new=copy.deepcopy(old);new['score']=.7
    new['cases']['horse_walk'].update(score=.4,survival=.5,failures=1)
    decision=compare(old,new)
    assert not decision['promote']
    assert 'horse_walk: more failures' in decision['reasons']


def test_equal_or_unpaired_reports_cannot_promote():
    old=_report()
    assert not compare(old,old)['promote']
    new=copy.deepcopy(old);new['seeds']=[999]
    with pytest.raises(ValueError, match='unpaired'):
        compare(old,new)


def test_stable_improvement_can_promote():
    old=_report();new=copy.deepcopy(old);new['score']=.51
    for c in new['cases'].values(): c['score']+=.01
    assert compare(old,new)['promote']


@pytest.mark.parametrize("metric,value,reason", [
    ("tilt_p95_deg",40.,"tilt regressed"),
    ("facing_mean",.3,"facing regressed"),
    ("slip_p95_m_s",1.,"foot slip regressed or unmeasured"),
    ("slip_p95_m_s",None,"foot slip regressed or unmeasured"),
    ("contact_fraction",.1,"ground contact regressed"),
])
def test_score_gain_does_not_mask_less_realistic_motion(metric,value,reason):
    old=_report();new=copy.deepcopy(old);new['score']=.6
    new['cases']['dog_walk'][metric]=value
    assert f'dog_walk: {reason}' in compare(old,new)['reasons']


def test_bird_height_gate_compares_command_relative_height_error():
    old=_report();old['family']='bird'
    for c in old['cases'].values(): c['flight_height_error_spans']=.3
    new=copy.deepcopy(old);new['score']=.7
    new['cases']['dog_walk']['flight_height_error_spans']=2.
    assert not compare(old,new)['promote']


def test_amp_critic_cannot_resume_with_different_data_or_reordered_segments(tmp_path):
    from motor.amp import Discriminator
    import torch
    torch.set_num_threads(1)
    first=tmp_path/'first.npz'; second=tmp_path/'second.npz'
    x=np.random.default_rng(4).normal(size=(20,8)).astype(np.float32)
    schema=np.array([['a','b','c'],['d','e','f']])
    np.savez(first,X=x,segments=schema,hz=30)
    a=Discriminator(first,hidden=8);state=a.state()
    b=Discriminator(first,hidden=8);b.load(state)
    np.savez(second,X=x,segments=schema[::-1],hz=30)
    with pytest.raises(ValueError,match='reference/schema'):
        Discriminator(second,hidden=8).load(state)
    legacy=dict(state);legacy.pop('reference')
    with pytest.raises(ValueError,match='legacy'):
        b.load(legacy)


def test_bird_curriculum_changes_training_body_without_changing_family_dimensions():
    from motor.bird import BirdEnv
    from bodies import BIRDS
    standard=BirdEnv(seed=5)
    warm=BirdEnv(seed=5,training_species='gull')
    assert warm.default_body()==BIRDS['gull']
    assert warm.sample_body(np.random.default_rng(4),0.).name=='gull'
    assert (warm.obs_dim,warm.act_dim)==(standard.obs_dim,standard.act_dim)
    assert standard.default_body()==BIRDS['crow']
    with pytest.raises(ValueError,match='unsupported'):
        BirdEnv(training_species='penguin')


def test_promotion_cannot_accumulate_realism_regressions():
    from motor.finetune import select_candidate
    initial=_report();champ=copy.deepcopy(initial);champ['score']=.6
    for c in champ['cases'].values():c['tilt_p95_deg']=14.9
    candidate=copy.deepcopy(champ);candidate['score']=.7
    for c in candidate['cases'].values():c['tilt_p95_deg']=19.8
    assert compare(champ,candidate)['promote']
    assert not select_candidate(initial,champ,candidate)['promote']


def test_final_holdout_cannot_overlap_validation():
    from motor.finetune import validate_seeds
    validate_seeds((101,202,303))
    for seeds in [(101,404),(404,505,606),(101,101),()]:
        with pytest.raises(ValueError,match='disjoint'):
            validate_seeds(seeds)


def test_rejected_round_does_not_seed_the_next_training_round(tmp_path,monkeypatch):
    import sys
    import motor.benchmark as benchmark
    import motor.show as show
    import motor.finetune as tune
    monkeypatch.chdir(tmp_path)
    initial=tmp_path/'initial.pt';initial.write_bytes(b'initial')
    monkeypatch.setattr(show,'load',lambda _p:({'family':'quadruped'},None))
    def fake_evaluate(p,*args):
        r=_report()
        if 'round' in str(p):
            r['score']=.7;r['cases']['horse_walk']['survival']=.1
        return r
    monkeypatch.setattr(benchmark,'evaluate',fake_evaluate)
    commands=[]
    def fake_run(cmd,**kw):
        commands.append(cmd)
        name=cmd[cmd.index('--name')+1]
        cp=tmp_path/'runs/motor'/name/'ckpt.pt';cp.parent.mkdir(parents=True)
        cp.write_bytes(b'candidate')
    monkeypatch.setattr(tune.subprocess,'run',fake_run)
    monkeypatch.setattr(tune.os,'nice',lambda _n:19)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BIPED_MUSCLES','QUAD_MUSCLES'):
        monkeypatch.setenv(name,'1')
    monkeypatch.setattr(sys,'argv',['finetune','--family','quadruped','--init',str(initial),
                                  '--name','mock_run','--rounds','2','--minutes','.01'])
    tune.main()
    assert len(commands)==2
    assert commands[0][commands[0].index('--init')+1]==commands[1][commands[1].index('--init')+1]
    import json
    result=json.loads((tmp_path/'runs/finetune/mock_run/history.json').read_text())
    assert result['champion'].endswith('baseline.pt')
    assert not (tmp_path/'runs/finetune/mock_run/active.lock').exists()


def test_single_species_prior_never_updates_from_other_bodies():
    from motor.ppo import motion_prior_rewards
    class Critic:
        def reward(self,x):return np.ones(len(x)),1.
        def update(self,x):self.seen=x.copy();return (1.,-1.)
    d=Critic();x=np.arange(24,dtype=np.float32).reshape(6,4)
    mask=np.array([True,False,False,True,False,False])
    styles,score,_=motion_prior_rewards(d,[x],[mask])
    np.testing.assert_array_equal(d.seen,x[mask])
    np.testing.assert_array_equal(styles[0],mask.astype(float))
    assert score==1.
    d=Critic();styles,score,_=motion_prior_rewards(d,[x],[np.zeros(6,bool)])
    assert not hasattr(d,'seen') and np.isnan(score)
    assert not np.any(styles[0])
