"""Task reward specification, independent of the simulation backend."""
GAMMA = 0.9997
GAE_LAMBDA = 0.999
FINISH = 100.0
FAILURE = -100.0
TIME_COST_PER_SECOND = 0.1
POTENTIAL_SCALE = 20.0

def scalar_reward(old_progress, progress, lap_length, old_done, done, success, dt=.1):
    if old_done:return 0.
    before=POTENTIAL_SCALE*max(0.,min(1.,old_progress/lap_length))
    after=0. if done else POTENTIAL_SCALE*max(0.,min(1.,progress/lap_length))
    terminal=(FINISH if success else FAILURE) if done else 0.
    return -TIME_COST_PER_SECOND*dt + GAMMA*after-before+terminal

def self_test():
    def episode(steps,success,forward=True):
        total=0.;old=0.
        for i in range(steps):
            new=(i+1)/steps*60 if forward else 0.
            total+=GAMMA**i*scalar_reward(old,new,60,False,i==steps-1,success)
            old=new
        return total
    assert episode(3000,True)>episode(3500,True)>episode(3500,False)
    assert episode(3500,True)>episode(1,False)
    assert episode(3500,False)<0
    assert scalar_reward(30,0,60,True,True,False)==0
    # Potential shaping cannot create net discounted reward by oscillating positions.
    a=scalar_reward(0,1,60,False,False,False)
    b=scalar_reward(1,0,60,False,True,False)
    assert abs(a+GAMMA*b-(-.01+GAMMA*(-.01+FAILURE)))<1e-10
    return dict(fast_clean_return=episode(3000,True),slow_clean_return=episode(3500,True),timeout_return=episode(3600,False),early_failure_return=episode(1,False))
if __name__=='__main__':print(self_test())
