import { register } from './ShowController';
register('particles', () => import('./modes/particles'));
register('portal',    () => import('./modes/portal'));
register('anamorph',  () => import('./modes/anamorph'));
register('glyphs',    () => import('./modes/glyphs'));
register('penrose',   () => import('./modes/penrose'));
