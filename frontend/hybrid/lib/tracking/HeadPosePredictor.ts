// ${IRIS_ROOT}\frontend\hybrid\lib\tracking\HeadPosePredictor.ts
import { OneEuro } from './OneEuroFilter';
import type { Pose, PredictedPose, PoseVec, Euler } from './types';

type Mode = 'ab' | 'abg';
export type PredictorParams = Partial<PredictorOpts>;

export interface PredictorOpts {
  mode?: Mode; euroMinCutoff?: number; euroBeta?: number; euroDerivCutoff?: number;
  alpha?: number; beta?: number; gamma?: number; maxDt?: number; confidenceFloor?: number;
}

export class HeadPosePredictor {
  private opts: Required<PredictorOpts>;
  private euroP = [new OneEuro(), new OneEuro(), new OneEuro()];
  private euroR = [new OneEuro(1.0, 0.05), new OneEuro(1.0, 0.05), new OneEuro(1.0, 0.05)];
  private x: PoseVec = [0,0,0]; private v: PoseVec = [0,0,0]; private a: PoseVec = [0,0,0];
  private rpy: Euler = [0,0,0]; private tPrev = 0; private initialized = false;
  private lastPrediction: PredictedPose = { p:[0,0,0], r:[0,0,0], confidence:0, t:0, dtAhead:0 };

  constructor(opts: PredictorOpts = {}) {
    this.opts = {
      mode: opts.mode ?? 'ab',
      euroMinCutoff: opts.euroMinCutoff ?? 1.0,
      euroBeta: opts.euroBeta ?? 0.2,
      euroDerivCutoff: opts.euroDerivCutoff ?? 1.0,
      alpha: opts.alpha ?? 0.8, beta: opts.beta ?? 0.4, gamma: opts.gamma ?? 0.05,
      maxDt: opts.maxDt ?? 50, confidenceFloor: opts.confidenceFloor ?? 0.25,
    };
  }

  reset(pose: Pose) {
    this.x = [...pose.p]; this.v = [0,0,0]; this.a = [0,0,0]; this.rpy = [...pose.r];
    this.tPrev = pose.t; this.initialized = true;
    this.lastPrediction = { p:[...this.x], r:[...this.rpy], confidence:pose.confidence, t:pose.t, dtAhead:0 };
  }

  update(meas: Pose, dtAheadMs: number): PredictedPose {
    if (!this.initialized) this.reset(meas);
    const dtMs = Math.min(this.opts.maxDt, Math.max(1e-3, meas.t - this.tPrev)); this.tPrev = meas.t;
    const dt = dtMs / 1000;

    // 1 Euro denoise
    const mP: PoseVec = [0,0,0]; const mR: Euler = [0,0,0];
    for (let i=0;i<3;i++) {
      this.euroP[i].minCutoff = this.opts.euroMinCutoff;
      this.euroP[i].beta = this.opts.euroBeta;
      this.euroP[i].dCutoff = this.opts.euroDerivCutoff;
      mP[i] = this.euroP[i].update(meas.p[i], meas.t);
      mR[i] = this.euroR[i].update(meas.r[i], meas.t);
    }

    if (meas.confidence < this.opts.confidenceFloor) {
      this.v = this.v.map(v => v * 0.9) as PoseVec; // freeze/decay velocity
      this.rpy = mR as Euler;
      return this.predict(dtAheadMs);
    }

    // alpha-beta[-gamma]
    if (this.opts.mode === 'ab') {
      const xbar: PoseVec = [ this.x[0] + this.v[0]*dt, this.x[1] + this.v[1]*dt, this.x[2] + this.v[2]*dt ];
      const r: PoseVec = [ mP[0]-xbar[0], mP[1]-xbar[1], mP[2]-xbar[2] ];
      this.x = [ xbar[0] + this.opts.alpha*r[0], xbar[1] + this.opts.alpha*r[1], xbar[2] + this.opts.alpha*r[2] ];
      this.v = [ this.v[0] + (this.opts.beta/dt)*r[0], this.v[1] + (this.opts.beta/dt)*r[1], this.v[2] + (this.opts.beta/dt)*r[2] ];
    } else {
      const xbar: PoseVec = [
        this.x[0] + this.v[0]*dt + 0.5*this.a[0]*dt*dt,
        this.x[1] + this.v[1]*dt + 0.5*this.a[1]*dt*dt,
        this.x[2] + this.v[2]*dt + 0.5*this.a[2]*dt*dt
      ];
      const vbar: PoseVec = [ this.v[0] + this.a[0]*dt, this.v[1] + this.a[1]*dt, this.v[2] + this.a[2]*dt ];
      const r: PoseVec = [ mP[0]-xbar[0], mP[1]-xbar[1], mP[2]-xbar[2] ];
      this.x = [ xbar[0] + this.opts.alpha*r[0], xbar[1] + this.opts.alpha*r[1], xbar[2] + this.opts.alpha*r[2] ];
      this.v = [ vbar[0] + (this.opts.beta/dt)*r[0], vbar[1] + (this.opts.beta/dt)*r[1], vbar[2] + (this.opts.beta/dt)*r[2] ];
      this.a = [ this.a[0] + 2*this.opts.gamma/(dt*dt)*r[0],
                 this.a[1] + 2*this.opts.gamma/(dt*dt)*r[1],
                 this.a[2] + 2*this.opts.gamma/(dt*dt)*r[2] ];
    }

    this.rpy = mR as Euler;
    return this.predict(dtAheadMs);
  }

  predict(dtAheadMs: number): PredictedPose {
    const dtA = Math.max(0, dtAheadMs) / 1000;
    const p: PoseVec = this.opts.mode === 'ab'
      ? [ this.x[0] + this.v[0]*dtA, this.x[1] + this.v[1]*dtA, this.x[2] + this.v[2]*dtA ]
      : [ this.x[0] + this.v[0]*dtA + 0.5*this.a[0]*dtA*dtA,
          this.x[1] + this.v[1]*dtA + 0.5*this.a[1]*dtA*dtA,
          this.x[2] + this.v[2]*dtA + 0.5*this.a[2]*dtA*dtA ];
    this.lastPrediction = { p, r:[...this.rpy], confidence:1, t:performance.now(), dtAhead: dtAheadMs };
    return this.lastPrediction;
  }

  peek(): PredictedPose { return this.lastPrediction; }
  
  setParams(p: PredictorParams) {
    this.opts = { ...this.opts, ...p };
  }
  
  getParams(): PredictorOpts {
    return { ...this.opts };
  }
}
