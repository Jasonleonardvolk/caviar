// ${IRIS_ROOT}\frontend\hybrid\lib\sensors\ISensorProvider.ts
import type { Pose } from '../tracking/types';

export interface ISensorProvider {
  start(onPose: (pose: Pose) => void): Promise<void>;
  stop(): void;
  isSupported(): Promise<boolean> | boolean;
  name: string;
}
