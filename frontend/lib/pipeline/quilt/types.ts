export type DisplayCalib = {
  pitch: number;   // lenticular pitch (px/view, includes DPI & view count scaling)
  tilt: number;    // vertical tilt factor
  center: number;  // view center/offset (pixels)
  subp: number;    // subpixel stride (0=R, 1=G, 2=B)
  panelW: number;  // physical panel width (px)
  panelH: number;  // physical panel height (px)
};

export type QuiltLayout = {
  cols: number;
  rows: number;
  tileW: number;
  tileH: number;
  numViews: number; // cols*rows recommended but not required
};