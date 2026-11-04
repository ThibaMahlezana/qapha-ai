import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

/** Backend started with: python -m uvicorn app.main:app --port 8000 */
export const API_BASE = '';

export type ModelName =
  | 'persistence'
  | 'linear'
  | 'adaptive'
  | 'policy'
  | 'constant_velocity_2pt'
  | 'alpha_beta'
  | 'kalman_cv'
  | 'bounce_aware';

export interface ModelStats {
  mean_error_px: number;
  maximum_error_px: number;
  errors_over_25px: number;
}

export interface Prediction {
  xy: [number, number];
  error_px: number;
}

export interface TimelineFrame {
  frame: number;
  observed: { bbox_xyxy_px: number[]; centre_xy: [number, number] };
  predictions: Record<ModelName, Prediction> | null;
  adaptive?: {
    selection: 'persistence' | 'linear';
    scores_before_observation: Record<string, number | null>;
    prediction_sha256: string;
  };
  policy?: { warning: boolean; override_active: boolean; selection: string };
  bounce?: {
    floor_bottom_px: number | null;
    p_bounce: number | null;
    hypotheses: {
      name: string;
      weight: number;
      centre_xy: [number, number];
      rx_px: number;
      ry_px: number;
      angle_deg: number;
    }[];
  };
  zone?: {
    model: ModelName;
    level: string;
    centre_xy: [number, number];
    rx_px: number;
    ry_px: number;
    angle_deg: number;
    observed_inside: boolean;
    nll: number;
  };
}

export interface Timeline {
  schema: string;
  source: { concept: string; label?: string; object_id: string; tracks_file?: string; tracks_sha256?: string };
  video: { file: string; fps: number; resolution_px: [number, number]; frame_offset?: number } | null;
  frame_count: number;
  models: ModelName[];
  summary: Record<ModelName, ModelStats>;
  warning_frames: number[];
  zone_summary?: { model: ModelName; coverage: Record<string, number>; mean_nll: number };
  final_learning_state_sha256: string;
  frames: TimelineFrame[];
  limitations: string[];
}

export interface TimelineListItem {
  name: string;
  frame_count: number;
  concept: string;
  video: Timeline['video'];
  summary: Record<ModelName, ModelStats>;
}

@Injectable({ providedIn: 'root' })
export class LabApiService {
  private http = inject(HttpClient);

  list(): Observable<TimelineListItem[]> {
    return this.http.get<TimelineListItem[]>(`${API_BASE}/api/timelines`);
  }

  timeline(name: string): Observable<Timeline> {
    return this.http.get<Timeline>(`${API_BASE}/api/timelines/${name}`);
  }

  frameUrl(name: string, index: number): string {
    return `${API_BASE}/api/timelines/${name}/frames/${index}`;
  }
}
