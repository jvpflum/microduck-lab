"""Preview-residual PPO in the original source-MJCF MuJoCo-Warp environment."""
import argparse
import copy
import gc
import json
import hashlib
import math
import os
import time
from pathlib import Path

from race_env import (torch, np, mujoco, wp, warnings, ort, Policy, Course, Driver,
                      BASE, SHA, TERM_DIMS, pack_history)
from v89_gpu_env import mw, t
from reward_spec import GAMMA, GAE_LAMBDA, self_test
from qualification import compare

ANCHOR_SHA = '2d1a028358803e845e58c10d22bcb80b4c9ae343f679e2b4c4d56846b2cfe3cd'


def write_json(name, value):
    target = Path(name); temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


class PreviewCourse(Course):
    """Original strict course, with extra deployable route preview and reset noise."""
    def __init__(self, n, policy, cfg):
        self.reset_counter = 0
        self.fixed_reset = None
        super().__init__(n, policy, cfg)

    def observe(self):
        base, throttle, steering, off = super().observe()
        xy = self.robot.root_link_pos_w[:, :2]
        q = self.robot.root_link_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        c, s = yaw.cos(), yaw.sin()
        future = []
        headings = torch.atan2(self.segments[:, 1], self.segments[:, 0])
        turns = []
        for offset in (4, 8, 12, 20):
            idx = (self.index + offset).clamp_max(len(self.points) - 1)
            delta = self.points[idx] - xy
            future += [(c * delta[:, 0] + s * delta[:, 1]) / 6,
                       (-s * delta[:, 0] + c * delta[:, 1]) / 6]
            before = (idx - 2).clamp(0, len(headings) - 1)
            after = (idx - 1).clamp(0, len(headings) - 1)
            angle = headings[after] - headings[before]
            turns.append(torch.atan2(angle.sin(), angle.cos()) / math.pi)
        extra = torch.stack(future + turns + [self.cmd[:, 0], self.cmd[:, 2] / 1.2], -1)
        return torch.cat([base, extra], -1), throttle, steering, off

    def reset(self):
        super().reset()
        self.driving = False
        if self.fixed_reset is None:
            seed = self.cfg['seed'] * 10000 + self.reset_counter
            heading, joint_noise = 3., .003
        else:
            seed, heading, joint_noise = self.fixed_reset
        generator = torch.Generator(device='cuda').manual_seed(seed)
        qp = t(self.e.d.qpos)
        qp[:, self.e.qpi] += (2 * torch.rand((self.n, 14), generator=generator, device='cuda') - 1) * joint_noise
        if self.fixed_reset is None:
            angle = (2 * torch.rand(self.n, generator=generator, device='cuda') - 1) * math.radians(heading)
        else:
            angle = torch.full((self.n,), math.radians(heading), device='cuda')
        qp[:, 3:7] = torch.stack([torch.cos(angle / 2), torch.zeros_like(angle),
                                  torch.zeros_like(angle), torch.sin(angle / 2)], -1)
        mw.forward(self.e.gm, self.e.d); wp.synchronize()
        self.previous.zero_(); self.cmd.zero_()
        self.history = self.frame()[:, None].repeat(1, 256, 1)
        self.initial_state_sha256 = hashlib.sha256(qp.cpu().numpy().tobytes()).hexdigest()
        self.reset_counter += 1
        self.driving = True
        self.probe(self.e)
        return self.observe()[0]


class PreviewDriver(torch.nn.Module):
    def __init__(self, anchor):
        super().__init__()
        self.anchor = copy.deepcopy(anchor).eval()
        for parameter in self.anchor.parameters(): parameter.requires_grad_(False)
        self.residual = torch.nn.Sequential(torch.nn.Linear(28, 128), torch.nn.Tanh(),
                                            torch.nn.Linear(128, 128), torch.nn.Tanh(),
                                            torch.nn.Linear(128, 2))
        self.critic = torch.nn.Sequential(torch.nn.Linear(28, 128), torch.nn.Tanh(),
                                          torch.nn.Linear(128, 128), torch.nn.Tanh(),
                                          torch.nn.Linear(128, 1))
        for module in (self.residual[-1], self.critic[-1]):
            torch.nn.init.zeros_(module.weight); torch.nn.init.zeros_(module.bias)
        self.register_buffer('bounds', torch.tensor([.4, .5]))

    def forward(self, x):
        with torch.no_grad(): dist, baseline_value = self.anchor(x[:, :14])
        mean = dist.mean + self.bounds * self.residual(x).tanh()
        return torch.distributions.Normal(mean, torch.full_like(mean, .035)), baseline_value + self.critic(x).squeeze(-1)


class AnchorView(torch.nn.Module):
    def __init__(self, anchor):
        super().__init__(); self.anchor = anchor

    def forward(self, x):
        return self.anchor(x[:, :14])


@torch.no_grad()
def evaluation(course, driver, label, seed, heading, steps=None, verify=None):
    course.fixed_reset = (seed, heading, .005)
    course.live.phase = label
    obs = course.reset()
    for _ in range(steps or int(course.cfg['episode_s'] * 10)):
        obs, _, _, _ = course.step(driver(obs)[0].mean, verify)
        if bool(course.done.all()): break
    result = course.result()
    corners = result.pop('corner_speeds')
    result.update(reset_sha256=course.initial_state_sha256, reset_seed=seed,
                  reset_heading_degrees=heading,
                  falls=int((course.tilt > math.pi / 4).sum()),
                  finite_failures=int((~course.probe.finite).sum()),
                  case_flags=dict(clean=course.success.cpu().tolist(),
                                  offtrack=course.offtrack.cpu().tolist(),
                                  body_contact=course.body.cpu().tolist()),
                  corner_speed_summary=dict(count=sum(len(r['speeds']) for r in corners),
                                            note='Raw corner records remain in worker scratch'))
    write_json(label + '.json', result)
    torch.save(torch.stack(course.traces), label + '-trace.pt')
    print(json.dumps(dict(stage=label, clean=result['clean_finishes'],
                          worlds=course.n, offtrack=result['offtrack_failures'],
                          body=result['body_contacts'])), flush=True)
    return result


def learn_update(course, driver, actor_optimizer, critic_optimizer, horizon):
    obs = course.reset(); rows = []
    for _ in range(horizon):
        active = course.step_count >= 100
        with torch.no_grad():
            dist, value = driver(obs); action = dist.sample(); logp = dist.log_prob(action).sum(-1)
        new_obs, reward, done, valid = course.step(action)
        rows.append((obs, action, logp, value, reward, done, valid & active, valid))
        obs = new_obs
        if bool(done.all()): break
    with torch.no_grad(): next_value = driver(obs)[1]
    adv = torch.zeros(course.n, device='cuda'); advantages = []
    for row in reversed(rows):
        mask = (~row[5]).float()
        adv = row[4] + GAMMA * next_value * mask - row[3] + GAMMA * GAE_LAMBDA * mask * adv
        advantages.append(adv); next_value = row[3]
    advantage = torch.stack(advantages[::-1])
    returns = (advantage + torch.stack([r[3] for r in rows])).flatten().detach()
    flatten = lambda i: torch.stack([r[i] for r in rows]).flatten(0, 1)
    x, action, old_logp = flatten(0), flatten(1), flatten(2)
    actor_mask, critic_mask = flatten(6), flatten(7)
    advantage = advantage.flatten().detach()
    assert bool(actor_mask.any()), 'No actor-controlling samples'
    advantage = (advantage - advantage[actor_mask].mean()) / (advantage[actor_mask].std(unbiased=False) + 1e-6)
    assert torch.isfinite(returns).all() and torch.isfinite(advantage).all()
    stop_actor = False; kls = []; actor_updates = 0
    for _ in range(3):
        for ids in torch.randperm(len(x), device='cuda').split(2048):
            critic_ids = ids[critic_mask[ids]]
            if len(critic_ids):
                value = driver(x[critic_ids])[1]
                loss = torch.nn.functional.smooth_l1_loss(value, returns[critic_ids])
                critic_optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(driver.critic.parameters(), .5, error_if_nonfinite=True)
                critic_optimizer.step()
            actor_ids = ids[actor_mask[ids]]
            if not len(actor_ids) or stop_actor: continue
            dist, _ = driver(x[actor_ids]); logp = dist.log_prob(action[actor_ids]).sum(-1)
            delta = logp - old_logp[actor_ids]; ratio = delta.exp()
            kl = float(((ratio - 1) - delta).mean().detach()); kls.append(kl)
            if kl > .015: stop_actor = True; continue
            with torch.no_grad(): anchor_mean = driver.anchor(x[actor_ids, :14])[0].mean
            loss = -torch.minimum(ratio * advantage[actor_ids], ratio.clamp(.92, 1.08) * advantage[actor_ids]).mean()
            loss += .3 * (dist.mean.tanh() - anchor_mean.tanh()).square().mean()
            assert torch.isfinite(loss)
            actor_optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(driver.residual.parameters(), .5, error_if_nonfinite=True)
            actor_optimizer.step(); actor_updates += 1
    result = course.result()
    return dict(driver_steps=len(rows), active_actor_samples=int(actor_mask.sum()),
                actor_updates=actor_updates, kl_max=max(kls, default=0),
                clean_finishes=result['clean_finishes'], offtrack_failures=result['offtrack_failures'],
                body_contacts=result['body_contacts'], mean_progress_m=result['mean_progress_m'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    manifest = json.loads(Path('campaign.json').read_text())
    cfg = json.loads(Path('course.json').read_text()); cfg['seed'] = args.seed
    torch.set_num_threads(2); torch.manual_seed(args.seed); torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available() and wp.get_device('cuda:0').is_cuda
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == SHA == manifest['V89_sha256']
    assert hashlib.sha256(Path('anchor.pt').read_bytes()).hexdigest() == ANCHOR_SHA
    self_test()
    anchor = Driver().cuda(); anchor.critic_net = copy.deepcopy(anchor.net)
    anchor.load_state_dict(torch.load('anchor.pt', map_location='cuda', weights_only=False)['driver'])
    anchor.eval()
    driver = PreviewDriver(anchor).cuda()
    test = torch.randn(64, 28, device='cuda')
    with torch.no_grad():
        assert torch.equal(driver(test)[0].mean, anchor(test[:, :14])[0].mean)
        delta = driver.bounds * driver.residual(test).tanh()
        assert (delta.abs() <= driver.bounds).all()
    initial = {k: v.detach().clone() for k, v in driver.residual.state_dict().items()}
    n = 8 if args.smoke else manifest['worlds']
    raw = Policy(BASE, 'cuda').eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        policy = torch.jit.trace(raw, torch.zeros(n, 15616, device='cuda'), check_trace=False)
    course = PreviewCourse(n, policy, cfg)
    actor_optimizer = torch.optim.Adam(driver.residual.parameters(), lr=3e-5)
    critic_optimizer = torch.optim.Adam(driver.critic.parameters(), lr=1e-4)
    runtime = dict(gpu=torch.cuda.get_device_name(0), physics_device='cuda:0', policy_device='cuda:0',
                   optimizer_device='cuda:0', torch=torch.__version__, mujoco=mujoco.__version__,
                   warp=wp.__version__, baseline_sha256=SHA, anchor_sha256=ANCHOR_SHA)
    write_json('runtime.json', runtime)
    if args.smoke:
        reference = ort.InferenceSession(str(BASE), providers=['CPUExecutionProvider'])
        evaluation(course, AnchorView(anchor), 'smoke-reference', 99099, 0, 80, reference)
        assert course.probe.finite.all() and not course.body.any()
        assert float(course.progress.max()) > .25
        course.fixed_reset = None
        row = learn_update(course, driver, actor_optimizer, critic_optimizer, 128)
        changes = {k: float((v - initial[k]).abs().max()) for k, v in driver.residual.state_dict().items()}
        assert row['actor_updates'] > 0 and changes['4.weight'] > 0 and changes['0.weight'] > 0
        assert hashlib.sha256(BASE.read_bytes()).hexdigest() == SHA
        write_json('summary.json', dict(status='smoke-passed', actual_learning=row,
                   parameter_change=changes, initial_anchor_action_error=0., runtime=runtime,
                   promotion=False, smoke_is_qualification=False))
        return
    started = time.time(); course.fixed_reset = None
    for update in range(1, manifest['updates'] + 1):
        course.live.phase = 'preview residual PPO training'; course.live.update = update
        row = learn_update(course, driver, actor_optimizer, critic_optimizer, cfg['horizon'])
        row.update(update=update, elapsed_s=time.time() - started, seed=args.seed)
        with Path('progress.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
        write_json('progress.json', row); print(json.dumps(row), flush=True)
        if update in manifest['checkpoint_updates']:
            torch.save(dict(driver=driver.state_dict(), actor_optimizer=actor_optimizer.state_dict(),
                            critic_optimizer=critic_optimizer.state_dict(), config=cfg,
                            manifest=manifest, update=update, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state()), f'driver-{update}.pt')
    del course; gc.collect(); torch.cuda.empty_cache()
    # Fixed independent reset rosters. Every saved checkpoint is compared; no
    # selected intermediate training metric substitutes for evaluation.
    evaluation_course = PreviewCourse(64, policy, cfg)
    controls = [evaluation(evaluation_course, AnchorView(anchor), f'selection-anchor-{i}',
                           cohort['seed'], cohort['heading_degrees'])
                for i, cohort in enumerate(manifest['selection'])]
    selection = []; eligible = []
    for update in manifest['checkpoint_updates']:
        driver.load_state_dict(torch.load(f'driver-{update}.pt', map_location='cuda', weights_only=False)['driver'])
        checks = []
        for i, cohort in enumerate(manifest['selection']):
            candidate = evaluation(evaluation_course, driver, f'selection-{update}-{i}', cohort['seed'], cohort['heading_degrees'])
            assert candidate['reset_sha256'] == controls[i]['reset_sha256']
            checks.append(compare(controls[i], candidate))
        record = dict(update=update, comparisons=checks, eligible=all(c['passed'] for c in checks))
        selection.append(record)
        if record['eligible']: eligible.append(record)
    write_json('selection.json', selection)
    final = []; selected = None
    if eligible:
        chosen = min(eligible, key=lambda row: (max(c['paired_median_time_ratio'] for c in row['comparisons']), row['update']))
        selected = chosen['update']
        driver.load_state_dict(torch.load(f'driver-{selected}.pt', map_location='cuda', weights_only=False)['driver'])
        del evaluation_course; gc.collect(); torch.cuda.empty_cache()
        evaluation_course = PreviewCourse(48, policy, cfg)
        for i, cohort in enumerate(manifest['final']):
            control = evaluation(evaluation_course, AnchorView(anchor), f'final-anchor-{i}', cohort['seed'], cohort['heading_degrees'])
            candidate = evaluation(evaluation_course, driver, f'final-candidate-{i}', cohort['seed'], cohort['heading_degrees'])
            assert candidate['reset_sha256'] == control['reset_sha256']
            final.append(compare(control, candidate, cohort.get('stress_only', False)))
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == SHA
    assert hashlib.sha256(Path('anchor.pt').read_bytes()).hexdigest() == ANCHOR_SHA
    write_json('summary.json', dict(status='complete', seed=args.seed, runtime=runtime,
               elapsed_s=time.time() - started, selection=selection, selected_update=selected,
               final=final, racing_candidate_passed=bool(final) and all(c['passed'] for c in final),
               general_model_release_qualified=False,
               remaining_gate='Standalone and actual-course skate/trick/recovery/race/skate integration evaluation',
               baseline_unchanged=True, promotion=False))


if __name__ == '__main__':
    main()
