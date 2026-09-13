"""Command skills for frozen V89. Heading input is radians, sampled at 50 Hz.

The caller must retain the ONNX observation history and feed these commands
through the normal V89 observation builder. No joint targets or states change.
Simulation-tested only; hardware heading estimation has not been validated.
"""
import math

class TrickController:
    def __init__(self):
        self.active=False
        self.status='idle'

    def start(self, heading, direction=1, degrees=360):
        if self.active:raise RuntimeError('A trick is already active')
        if direction not in (-1,1) or degrees not in (180,360,720):raise ValueError('Invalid trick')
        if not math.isfinite(heading):raise ValueError('Nonfinite heading')
        self.previous=float(heading);self.progress=0.;self.elapsed=0.
        self.target=direction*math.radians(degrees);self.active=True;self.status='running'

    def cancel(self):
        self.active=False;self.status='cancelled'

    def update(self, heading, commands, dt=.02):
        result=list(commands)
        if not self.active:return result
        if len(result)!=13 or not math.isfinite(heading) or not 0<dt<=.05:
            raise ValueError('Expected finite heading, 13 commands and <=50ms update')
        delta=heading-self.previous
        self.progress+=math.atan2(math.sin(delta),math.cos(delta));self.previous=heading
        error=self.target-self.progress
        self.elapsed+=dt
        if self.elapsed>15.000001:
            self.active=False
            # Heading-only completion is not the full simulation qualification.
            self.status='heading_complete' if abs(error)<math.radians(20) else 'timeout'
            return result
        result[0]=.30
        result[1]=0.
        result[2]=0. if abs(error)<math.radians(5) else max(-.60,min(.60,.7*error))
        return result
