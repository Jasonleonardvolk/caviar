// ${IRIS_ROOT}\frontend\hybrid\lib\tracking\types.ts
export type PoseVec = [number, number, number]; // x,y,z (screen or meters)
export type Euler   = [number, number, number]; // yaw,pitch,roll (rad)

export interface Pose {
  p: PoseVec;
  r: Euler;
  confidence: number;   // 0..1
  t: number;            // ms (performance.now())
}

export interface PredictedPose extends Pose {
  dtAhead: number;      // ms predicted ahead
}
