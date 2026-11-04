import {
  Component, HostListener, OnDestroy, OnInit, computed, inject, signal,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';
import {
  LabApiService, ModelName, Timeline, TimelineFrame, TimelineListItem,
} from './lab-api.service';

interface ModelInfo { key: ModelName; label: string; colour: string; note: string }

export const MODEL_INFO: ModelInfo[] = [
  { key: 'bounce_aware', label: 'Bounce-aware (floor memory)', colour: '#0891b2',
    note: 'Learns the floor from bounces it has seen, then weighs "keeps falling" against "bounces"' },
  { key: 'kalman_cv', label: 'Kalman + 90% zone', colour: '#d97706',
    note: 'Constant-velocity Kalman filter; the ellipse is where it is 90% sure the ball will be' },
  { key: 'adaptive', label: 'Adaptive selector', colour: '#2563eb',
    note: 'Picks persistence or linear, whichever had lower recent error' },
  { key: 'policy', label: 'Warning policy', colour: '#9333ea',
    note: 'Adaptive, but switches to persistence for 3 frames after a warning' },
  { key: 'constant_velocity_2pt', label: 'Constant velocity', colour: '#059669',
    note: 'Repeats the last one-frame step' },
  { key: 'alpha_beta', label: 'Alpha-beta filter', colour: '#e11d48',
    note: 'Simple fixed-gain tracking filter' },
  { key: 'linear', label: 'Linear (6-frame)', colour: '#ea580c',
    note: 'Average velocity over the last 5 steps' },
  { key: 'persistence', label: 'Persistence', colour: '#64748b',
    note: 'Assumes the ball stays where it was' },
];

const CHART_W = 760;
const CHART_H = 170;
const CHART_PAD = { l: 40, r: 10, t: 10, b: 22 };

@Component({
  selector: 'qapha-lab',
  imports: [DecimalPipe],
  templateUrl: './lab.component.html',
})
export class LabComponent implements OnInit, OnDestroy {
  private api = inject(LabApiService);

  readonly models = MODEL_INFO;
  readonly chart = { w: CHART_W, h: CHART_H, pad: CHART_PAD };

  clips = signal<TimelineListItem[]>([]);
  clipName = signal<string>('');
  timeline = signal<Timeline | null>(null);
  index = signal(0);
  playing = signal(false);
  speed = signal(0.25);
  error = signal<string | null>(null);
  visible = signal<Set<ModelName>>(
    new Set<ModelName>(['bounce_aware', 'kalman_cv', 'constant_velocity_2pt']));

  private timer: ReturnType<typeof setInterval> | null = null;

  frame = computed<TimelineFrame | null>(() => {
    const t = this.timeline();
    return t ? t.frames[this.index()] : null;
  });

  size = computed<[number, number]>(
    () => this.timeline()?.video?.resolution_px ?? [1280, 720]);

  /** Retry counter per frame: a failed frame image is re-requested. */
  private retries = signal<Record<number, number>>({});
  private preloaded = new Set<string>();

  frameUrl = computed(() => {
    if (!this.clipName()) return '';
    const r = this.retries()[this.index()] ?? 0;
    const url = this.api.frameUrl(this.clipName(), this.index());
    return r ? `${url}?retry=${r}` : url;
  });

  /** Fit the picture inside ~64% of the window height, keeping its shape. */
  stageMaxWidth = computed(() => {
    const [w, h] = this.size();
    return `min(100%, calc(64vh * ${w} / ${h}))`;
  });

  trail = computed(() => {
    const t = this.timeline();
    if (!t) return '';
    const start = Math.max(0, this.index() - 10);
    return t.frames.slice(start, this.index() + 1)
      .map(f => f.observed.centre_xy.join(',')).join(' ');
  });

  rows = computed(() => {
    const f = this.frame();
    if (!f?.predictions) return [];
    return this.models.map(m => ({
      ...m, ...f.predictions![m.key], offFrame: this.offFrame(f.predictions![m.key].xy),
    }));
  });

  /** Visible predictions; ones outside the image are pinned to its edge. */
  markers = computed(() => {
    const f = this.frame();
    if (!f?.predictions) return [];
    const [w, h] = this.size();
    return this.models.filter(m => this.visible().has(m.key)).map(m => {
      const [x, y] = f.predictions![m.key].xy;
      return {
        key: m.key, colour: m.colour, x, y,
        px: Math.max(0, Math.min(w, x)), py: Math.max(0, Math.min(h, y)),
        off: this.offFrame([x, y]),
      };
    });
  });

  ranking = computed(() => {
    const t = this.timeline();
    if (!t) return [];
    return [...this.models]
      .map(m => ({ ...m, ...t.summary[m.key] }))
      .sort((a, b) => a.mean_error_px - b.mean_error_px);
  });

  yMax = computed(() => {
    const t = this.timeline();
    if (!t) return 100;
    const peak = Math.max(...this.models.map(m => t.summary[m.key].maximum_error_px));
    return Math.ceil(peak / 20) * 20;
  });

  chartLines = computed(() => {
    const t = this.timeline();
    if (!t) return [];
    return this.models
      .filter(m => this.visible().has(m.key))
      .map(m => ({
        colour: m.colour,
        points: t.frames.filter(f => f.predictions)
          .map(f => `${this.cx(f.frame)},${this.cy(f.predictions![m.key].error_px)}`)
          .join(' '),
      }));
  });

  /** 90% zone radius (major semi-axis) over time, for the chart. */
  zoneLine = computed(() => {
    const t = this.timeline();
    if (!t || !this.visible().has('kalman_cv')) return '';
    return t.frames.filter(f => f.zone)
      .map(f => `${this.cx(f.frame)},${this.cy(f.zone!.rx_px)}`).join(' ');
  });

  yTicks = computed(() => {
    const max = this.yMax();
    const step = max <= 60 ? 10 : max <= 160 ? 20 : 50;
    const ticks: number[] = [];
    for (let v = 0; v <= max; v += step) ticks.push(v);
    return ticks;
  });

  ngOnInit(): void {
    this.api.list().subscribe({
      next: clips => {
        this.clips.set(clips);
        if (clips.length) this.load(clips[clips.length - 1].name);
      },
      error: () => this.error.set(
        'Cannot reach the Qapha API. Start it with: python -m uvicorn app.main:app --port 8000'),
    });
  }

  ngOnDestroy(): void { this.stop(); }

  load(name: string): void {
    this.stop();
    this.clipName.set(name);
    this.timeline.set(null);
    this.api.timeline(name).subscribe({
      next: t => {
        this.timeline.set(t);
        this.index.set(Math.min(2, t.frame_count - 1));
        this.retries.set({});
        this.preloaded.clear();
        this.preloadFrom(0);
      },
      error: () => this.error.set(`Could not load timeline ${name}`),
    });
  }

  go(i: number): void {
    const t = this.timeline();
    if (!t) return;
    this.index.set(Math.max(0, Math.min(t.frame_count - 1, i)));
    this.preloadFrom(this.index());
  }

  /** Fetch only the next few frames, so the backend is never flooded. */
  private preloadFrom(start: number): void {
    const t = this.timeline();
    if (!t) return;
    for (let i = start; i < Math.min(start + 8, t.frame_count); i++) {
      const url = this.api.frameUrl(this.clipName(), i);
      if (!this.preloaded.has(url)) {
        this.preloaded.add(url);
        new Image().src = url;
      }
    }
  }

  onFrameError(): void {
    const i = this.index();
    const tries = this.retries()[i] ?? 0;
    if (tries >= 4) return;
    setTimeout(() => this.retries.update(r => ({ ...r, [i]: tries + 1 })),
               400 * (tries + 1));
  }

  toggle(key: ModelName): void {
    const next = new Set(this.visible());
    next.has(key) ? next.delete(key) : next.add(key);
    this.visible.set(next);
  }

  play(): void {
    const t = this.timeline();
    if (!t) return;
    if (this.index() >= t.frame_count - 1) this.index.set(0);
    this.playing.set(true);
    const fps = (t.video?.fps ?? 25) * this.speed();
    this.timer = setInterval(() => {
      if (this.index() >= t.frame_count - 1) { this.stop(); return; }
      this.go(this.index() + 1);
    }, 1000 / fps);
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.playing.set(false);
  }

  setSpeed(value: string): void {
    this.speed.set(Number(value));
    if (this.playing()) { this.stop(); this.play(); }
  }

  @HostListener('window:keydown', ['$event'])
  onKey(event: KeyboardEvent): void {
    if ((event.target as HTMLElement)?.tagName === 'SELECT') return;
    if (event.key === 'ArrowRight') { this.stop(); this.go(this.index() + 1); }
    else if (event.key === 'ArrowLeft') { this.stop(); this.go(this.index() - 1); }
    else if (event.key === ' ') {
      event.preventDefault();
      this.playing() ? this.stop() : this.play();
    }
  }

  chartClick(event: MouseEvent): void {
    const svg = event.currentTarget as SVGSVGElement;
    const box = svg.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * CHART_W;
    const t = this.timeline();
    if (!t) return;
    const span = CHART_W - CHART_PAD.l - CHART_PAD.r;
    this.stop();
    this.go(Math.round(((x - CHART_PAD.l) / span) * (t.frame_count - 1)));
  }

  cx(frame: number): number {
    const n = (this.timeline()?.frame_count ?? 2) - 1;
    return CHART_PAD.l + (frame / n) * (CHART_W - CHART_PAD.l - CHART_PAD.r);
  }

  cy(error: number): number {
    const v = Math.min(error, this.yMax());
    return CHART_PAD.t + (1 - v / this.yMax()) * (CHART_H - CHART_PAD.t - CHART_PAD.b);
  }

  offFrame([x, y]: [number, number]): boolean {
    const [w, h] = this.size();
    return x < 0 || y < 0 || x > w || y > h;
  }

  colour(key: string): string {
    return this.models.find(m => m.key === key)?.colour ?? '#fff';
  }

  label(key: string): string {
    return this.models.find(m => m.key === key)?.label ?? key;
  }

  /** Hypothesis zones of the bounce-aware model, faded by their weight. */
  hypotheses = computed(() => {
    const f = this.frame();
    if (!f?.bounce || !this.visible().has('bounce_aware')) return [];
    // Hide hypotheses with under 5% weight; they only add clutter.
    return f.bounce.hypotheses
      .map((h, i) => ({
        ...h,
        cls: i === 0 ? 'hyp-fall' : 'hyp-bounce',
        opacity: 0.35 + 0.65 * h.weight,
        label: `${h.name} ${Math.round(h.weight * 100)}%`,
        labelDy: i === 0 ? 24 : -14,
      }))
      .filter(h => h.weight >= 0.05);
  });

  short(hash?: string): string {
    return hash ? `${hash.slice(0, 10)}…` : '';
  }
}
