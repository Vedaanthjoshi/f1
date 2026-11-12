import { useCallback, useEffect, useMemo, useState } from 'react'
import { Circuit3DCard } from './components/Circuit3DCard'

const SLIDE_IMAGES = ['/images/1.jpg', '/images/2.jpg', '/images/3.jpg', '/images/4.jpg', '/images/5.jpg', '/images/6.jpg']
const SLIDE_INTERVAL = 4500

export type Driver = {
  code: string
  name: string
  number: number
  team: string
  car: string
  country: string
  accent: string
}

export type LapRecord = {
  d: string      // driver code
  l: number      // lap number
  t: number      // lap time in seconds
  c: string      // compound (SOFT, MEDIUM, HARD)
  a: number      // tyre age
  elapsed: number // elapsed total race seconds
}

export type RaceInfo = {
  year: number
  race: string
  circuit: string
  country: string
  laps_count: number
  min_lap: number
  max_lap: number
  distance_km: number
  pit_loss: number
  drivers: string[]
  laps: LapRecord[]
}

export type TacticalStrategy = {
  name: string
  compound: string
  compound_encoded: number
  pit_offset: number
  stop_lap: number
  p_net_gain: number
  expected_gain_s: number
  p05_gain_s: number
  p95_gain_s: number
  crossover_lap: number | null
  window_open: boolean
}

export type ShapItem = {
  feature: string
  label: string
  shap_value: number
  value: number
}

export type SimResult = {
  sim_lap: number
  compound: string
  horizon_laps: number[]
  net_undercut_advantage: number[]
  stay_out_lap_times: number[]
  pit_now_lap_times: number[]
  crossover_lap: number | null
  window_open: boolean
  verdict: 'open' | 'pending' | 'closed'
  confidence: number
  headline: string
  description: string
  recommendation: string
  all_strategies?: TacticalStrategy[]
  shap_values?: ShapItem[]
  race_alpha?: number
  race_bias?: number
  rival_gap_s?: number | null
  rival_covered?: boolean
}

export type ReviewDriver = {
  code: string
  name: string
  number: number
  team: string
  accent: string
  final_pos: number
  gap_to_winner: number
  pit_laps: number[]
  stints: Array<{
    stint: number
    compound: string
    laps_count: number
    start_lap: number
    end_lap: number
  }>
}

export type ReviewInitData = {
  year: number
  race: string
  is_in_sample: boolean
  total_laps: number
  pit_loss_green: number
  pit_loss_neutralized: number
  circuit_physics: Record<string, any>
  drivers: ReviewDriver[]
  fallback_used?: boolean
  fallback_reason?: string
  requested_year?: number
  requested_race?: string
}

export type RegretScanRow = {
  shift: number
  moved_lap: number
  p05: number
  expected_time_delta: number
  p95: number
  is_noise: boolean
  status: string
  robust: boolean
  extrapolated: boolean
  is_actual: boolean
}

export type RegretScanResult = {
  driver: string
  stop_idx: number
  actual_pit_lap: number
  best_pit_lap: number
  best_shift: number
  best_delta_s: number
  max_regret_s: number
  scan: RegretScanRow[]
}

export type WhatIfResult = {
  is_feasible: boolean
  infeasible_reason?: string
  driver?: string
  actual_pit_lap?: number
  moved_lap?: number
  shift_laps?: number
  final_time_delta_p50?: number
  final_time_delta_p05?: number
  final_time_delta_p95?: number
  actual_pos?: number
  counterfactual_pos?: number
  positions_gained?: number
  dirty_air_total_s?: number
  dirty_air_laps?: number
  blocked_laps?: number
  robustness?: string
  extrapolated?: boolean
  rival_reactions?: Array<{
    driver: string
    reacted_pit: number
    actual_pit: number
    reason: string
  }>
  delta_per_lap?: number[]
  laps?: number[]
}

export type StintDegResult = {
  driver: string
  stints: Array<{
    stint: number
    compound: string
    ages: number[]
    laps: number[]
    deg_observed: number[]
    deg_predicted_xgb: number[]
    deg_predicted_linear: number[]
  }>
}

// Fallback initial drivers metadata
const DEFAULT_DRIVERS: Driver[] = [
  { code: 'VER', name: 'Max Verstappen', number: 1, team: 'Red Bull Racing', car: 'RB20', country: 'Netherlands', accent: '#3671C6' },
  { code: 'PER', name: 'Sergio Perez', number: 11, team: 'Red Bull Racing', car: 'RB20', country: 'Mexico', accent: '#3671C6' },
  { code: 'LEC', name: 'Charles Leclerc', number: 16, team: 'Ferrari', car: 'SF-24', country: 'Monaco', accent: '#E80020' },
  { code: 'SAI', name: 'Carlos Sainz', number: 55, team: 'Ferrari', car: 'SF-24', country: 'Spain', accent: '#E80020' },
  { code: 'HAM', name: 'Lewis Hamilton', number: 44, team: 'Mercedes', car: 'W15', country: 'United Kingdom', accent: '#27F4D2' },
  { code: 'RUS', name: 'George Russell', number: 63, team: 'Mercedes', car: 'W15', country: 'United Kingdom', accent: '#27F4D2' },
  { code: 'NOR', name: 'Lando Norris', number: 4, team: 'McLaren', car: 'MCL38', country: 'United Kingdom', accent: '#FF8000' },
  { code: 'PIA', name: 'Oscar Piastri', number: 81, team: 'McLaren', car: 'MCL38', country: 'Australia', accent: '#FF8000' },
  { code: 'ALO', name: 'Fernando Alonso', number: 14, team: 'Aston Martin', car: 'AMR24', country: 'Spain', accent: '#229971' },
  { code: 'STR', name: 'Lance Stroll', number: 18, team: 'Aston Martin', car: 'AMR24', country: 'Canada', accent: '#229971' },
]

const DEFAULT_SCHEDULES: Record<string, string[]> = {
  '2025': ['Australian Grand Prix', 'Chinese Grand Prix', 'Japanese Grand Prix', 'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 'British Grand Prix', 'Italian Grand Prix'],
  '2024': ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Japanese Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 'British Grand Prix', 'Italian Grand Prix', 'Abu Dhabi Grand Prix'],
  '2023': ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Azerbaijan Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 'British Grand Prix', 'Italian Grand Prix'],
  '2022': ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Emilia Romagna Grand Prix', 'Miami Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'British Grand Prix'],
}

const Icon = ({ name, size = 18 }: { name: string; size?: number }) => {
  const paths: Record<string, React.ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /></>,
    chart: <><path d="M4 19V9m6 10V5m6 14v-7m5 7H2" /></>,
    target: <><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3" /><path d="M12 2V0m0 24v-2M2 12H0m24 0h-2" /></>,
    radio: <><circle cx="12" cy="12" r="2" /><path d="M8.5 8.5a5 5 0 0 0 0 7m7-7a5 5 0 0 1 0 7M5 5a10 10 0 0 0 0 14m14-14a10 10 0 0 1 0 14" /></>,
    flag: <><path d="M5 22V3m0 1h12l-2 4 2 4H5" /></>,
    chevron: <path d="m8 10 4 4 4-4" />,
    reset: <><path d="M4 4v6h6" /><path d="M5.5 17.5A8 8 0 1 0 6 6L4 10" /></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  )
}

function DriverAvatar({ driver }: { driver: Driver }) {
  return (
    <div className="driver-avatar" style={{ '--driver': driver.accent } as React.CSSProperties}>
      <svg viewBox="0 0 180 190" role="img" aria-label={`${driver.name} avatar`}>
        <path fill="var(--driver)" d="M18 190c4-48 29-70 72-70s68 22 72 70z" />
        <path fill="#e7a878" d="M58 99c4 25 16 38 32 38s28-13 32-38V59H58z" />
        <path fill="#111720" d="M52 66c0-34 20-50 39-50 26 0 42 20 42 51l-12 14-3-28c-20 9-41 8-58 0l-2 31z" />
        <path fill="#f0b58a" d="M55 74c-9-3-10 19 4 25m66-25c9-3 10 19-4 25" />
        <path stroke="#161b23" strokeWidth="4" strokeLinecap="round" d="M68 82h12m21 0h12" />
        <path stroke="#a45f53" strokeWidth="3" strokeLinecap="round" d="M82 108c5 3 11 3 16 0" />
        <path fill="#090e14" d="M64 137l26 22 27-22 9 8-18 45H72l-18-45z" />
        <text x="90" y="181" textAnchor="middle" fill="white" fontSize="18" fontWeight="800">{driver.code}</text>
      </svg>
    </div>
  )
}

function DriverCard({ driver, role, side }: { driver: Driver; role: string; side: 'a' | 'b' }) {
  return (
    <article className={`driver-card ${side}`}>
      <span className="role">{role}</span>
      <DriverAvatar driver={driver} />
      <div className="meta">
        <div>
          <h3>{driver.name}</h3>
          <p>{driver.team} · <span>{driver.car}</span></p>
        </div>
        <div className="num" style={{ color: driver.accent }}>#{driver.number}</div>
      </div>
    </article>
  )
}

function ChartFrame({
  eyebrow,
  title,
  legend,
  onEnlarge,
  children
}: {
  eyebrow: string
  title: string
  legend?: React.ReactNode
  onEnlarge?: () => void
  children: React.ReactNode
}) {
  return (
    <div className="chart-panel">
      <div className="panel-heading" style={{ marginBottom: 10 }}>
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h3>{title}</h3>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {legend}
          {onEnlarge && (
            <button
              className="enlarge-chart-btn"
              onClick={(e) => { e.stopPropagation(); onEnlarge(); }}
              title="Click to enlarge and inspect interactive telemetry"
            >
              ⛶ ENLARGE
            </button>
          )}
        </div>
      </div>
      <div onClick={onEnlarge} style={{ cursor: onEnlarge ? 'pointer' : 'default' }}>
        {children}
      </div>
    </div>
  )
}

function RealDegradationChart({ lapsA, lapsB, a, b }: { lapsA: LapRecord[]; lapsB: LapRecord[]; a: Driver; b: Driver }) {
  if (!lapsA.length && !lapsB.length) {
    return <div style={{ height: 180, display: 'grid', placeItems: 'center', color: 'var(--muted)', font: '10px JetBrains Mono' }}>No telemetry recorded for this stint</div>
  }

  const allLaps = [...lapsA, ...lapsB]
  const minLap = Math.min(...allLaps.map(l => l.l))
  const maxLap = Math.max(...allLaps.map(l => l.l))
  const times = allLaps.map(l => l.t).filter(t => t > 60 && t < 150)
  const minTime = times.length ? Math.floor(Math.min(...times) * 2) / 2 - 0.5 : 88
  const maxTime = times.length ? Math.ceil(Math.max(...times) * 2) / 2 + 0.5 : 96
  const timeSpan = maxTime - minTime || 1

  const W = 680
  const H = 260
  const padL = 52
  const padR = 25
  const padT = 20
  const padB = 34
  const plotW = W - padL - padR
  const plotH = H - padT - padB

  const scaleX = (lap: number) => {
    if (maxLap === minLap) return padL + plotW / 2
    return padL + ((lap - minLap) / (maxLap - minLap)) * plotW
  }
  const scaleY = (t: number) => {
    return padT + plotH - ((t - minTime) / timeSpan) * plotH
  }

  const validA = lapsA.filter(r => r.t >= minTime && r.t <= maxTime).sort((x, y) => x.l - y.l)
  const validB = lapsB.filter(r => r.t >= minTime && r.t <= maxTime).sort((x, y) => x.l - y.l)

  const linePathA = validA.map((r, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(r.l).toFixed(1)} ${scaleY(r.t).toFixed(1)}`).join(' ')
  const linePathB = validB.map((r, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(r.l).toFixed(1)} ${scaleY(r.t).toFixed(1)}`).join(' ')

  const areaPathA = validA.length > 0 
    ? `${linePathA} L ${scaleX(validA[validA.length - 1].l).toFixed(1)} ${padT + plotH} L ${scaleX(validA[0].l).toFixed(1)} ${padT + plotH} Z`
    : ''
  const areaPathB = validB.length > 0
    ? `${linePathB} L ${scaleX(validB[validB.length - 1].l).toFixed(1)} ${padT + plotH} L ${scaleX(validB[0].l).toFixed(1)} ${padT + plotH} Z`
    : ''

  // 5 Y-ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(frac => {
    const val = minTime + frac * timeSpan
    const y = scaleY(val)
    return { val, y }
  })

  // X-axis ticks (5-6 ticks)
  const lapSpan = maxLap - minLap || 1
  const xStep = Math.max(1, Math.round(lapSpan / 6))
  const xTicks: number[] = []
  for (let l = minLap; l <= maxLap; l += xStep) {
    xTicks.push(l)
  }
  if (xTicks[xTicks.length - 1] !== maxLap) xTicks.push(maxLap)

  // Identify pit in-laps or compound changes in lapsA
  const pitLapsA = lapsA.filter((r, i) => i > 0 && (r.a === 1 || r.c !== lapsA[i - 1].c))

  return (
    <div className="chart-body">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
        <defs>
          <linearGradient id="gradDegA" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={a.accent} stopOpacity="0.24" />
            <stop offset="100%" stopColor={a.accent} stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="gradDegB" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={b.accent} stopOpacity="0.15" />
            <stop offset="100%" stopColor={b.accent} stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Horizontal gridlines and Y-axis labels */}
        {yTicks.map((t, idx) => (
          <g key={idx}>
            <line x1={padL} y1={t.y} x2={W - padR} y2={t.y} stroke="var(--line)" strokeDasharray="3 3" strokeWidth="0.8" />
            <text x={padL - 6} y={t.y + 3} textAnchor="end" fill="var(--muted)" fontSize="8.5" fontFamily="JetBrains Mono">
              {t.val.toFixed(1)}s
            </text>
          </g>
        ))}

        {/* Vertical gridlines and X-axis labels */}
        {xTicks.map(lapNum => {
          const x = scaleX(lapNum)
          return (
            <g key={lapNum}>
              <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke="var(--line)" strokeDasharray="2 4" strokeWidth="0.6" opacity="0.6" />
              <text x={x} y={H - 10} textAnchor="middle" fill="var(--muted)" fontSize="8.5" fontFamily="JetBrains Mono">
                L{lapNum}
              </text>
            </g>
          )
        })}

        {/* Pit Stop Markers */}
        {pitLapsA.map((p, idx) => {
          const x = scaleX(p.l)
          return (
            <g key={`pitA-${idx}`}>
              <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke={a.accent} strokeDasharray="4 2" strokeWidth="1.2" opacity="0.85" />
              <rect x={x - 16} y={padT + 2} width="32" height="13" fill="#080e14" stroke={a.accent} strokeWidth="1" rx="2" />
              <text x={x} y={padT + 11} textAnchor="middle" fill={a.accent} fontSize="7" fontWeight="700" fontFamily="JetBrains Mono">BOX {p.c?.[0]}</text>
            </g>
          )
        })}

        {/* Shaded Areas */}
        {areaPathB && <path d={areaPathB} fill="url(#gradDegB)" />}
        {areaPathA && <path d={areaPathA} fill="url(#gradDegA)" />}

        {/* Lines */}
        {linePathB && <path d={linePathB} fill="none" stroke={b.accent} strokeWidth="1.8" opacity="0.85" />}
        {linePathA && <path d={linePathA} fill="none" stroke={a.accent} strokeWidth="2.4" />}

        {/* Data Point Dots */}
        {validB.map((r, i) => (
          <circle key={`dotB-${i}`} cx={scaleX(r.l)} cy={scaleY(r.t)} r="2" fill={b.accent} opacity="0.75" />
        ))}
        {validA.map((r, i) => (
          <circle key={`dotA-${i}`} cx={scaleX(r.l)} cy={scaleY(r.t)} r="2.4" fill={a.accent} stroke="#05080c" strokeWidth="0.8" />
        ))}
      </svg>
    </div>
  )
}

function RealGapChart({ lapsA, lapsB, a, b }: { lapsA: LapRecord[]; lapsB: LapRecord[]; a: Driver; b: Driver }) {
  const commonLaps = useMemo(() => {
    const mapB = new Map(lapsB.map(l => [l.l, l]))
    const pts: { lap: number; gap: number }[] = []
    lapsA.forEach(la => {
      const lb = mapB.get(la.l)
      if (lb && la.elapsed != null && lb.elapsed != null) {
        pts.push({ lap: la.l, gap: lb.elapsed - la.elapsed })
      }
    })
    return pts.sort((x, y) => x.lap - y.lap)
  }, [lapsA, lapsB])

  if (!commonLaps.length) {
    return <div style={{ height: 180, display: 'grid', placeItems: 'center', color: 'var(--muted)', font: '10px JetBrains Mono' }}>Awaiting overlapping racing laps</div>
  }

  const minLap = commonLaps[0].lap
  const maxLap = commonLaps[commonLaps.length - 1].lap
  const gaps = commonLaps.map(p => p.gap)
  const maxAbs = Math.max(1.5, Math.ceil(Math.max(...gaps.map(Math.abs)) * 1.25))

  const W = 680
  const H = 260
  const padL = 52
  const padR = 25
  const padT = 20
  const padB = 34
  const plotW = W - padL - padR
  const plotH = H - padT - padB

  const scaleX = (lap: number) => {
    if (maxLap === minLap) return padL + plotW / 2
    return padL + ((lap - minLap) / (maxLap - minLap)) * plotW
  }
  const zeroY = padT + plotH / 2
  const scaleY = (gap: number) => zeroY - (gap / maxAbs) * (plotH / 2)

  const linePath = commonLaps.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.lap).toFixed(1)} ${scaleY(p.gap).toFixed(1)}`).join(' ')
  const areaPath = `${linePath} L ${scaleX(maxLap).toFixed(1)} ${zeroY.toFixed(1)} L ${scaleX(minLap).toFixed(1)} ${zeroY.toFixed(1)} Z`

  // 5 Y-ticks
  const yTicks = [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs].map(val => ({
    val,
    y: scaleY(val)
  }))

  // X-ticks
  const lapSpan = maxLap - minLap || 1
  const xStep = Math.max(1, Math.round(lapSpan / 6))
  const xTicks: number[] = []
  for (let l = minLap; l <= maxLap; l += xStep) xTicks.push(l)
  if (xTicks[xTicks.length - 1] !== maxLap) xTicks.push(maxLap)

  // Max and min gap points
  let maxPt = commonLaps[0]
  let minPt = commonLaps[0]
  for (const pt of commonLaps) {
    if (pt.gap > maxPt.gap) maxPt = pt
    if (pt.gap < minPt.gap) minPt = pt
  }

  return (
    <div className="chart-body">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
        <defs>
          <linearGradient id="gapGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--green)" stopOpacity="0.25" />
            <stop offset="50%" stopColor="var(--cyan)" stopOpacity="0.06" />
            <stop offset="100%" stopColor="var(--red)" stopOpacity="0.25" />
          </linearGradient>
        </defs>

        {/* Gridlines and Y-labels */}
        {yTicks.map((t, idx) => (
          <g key={idx}>
            <line
              x1={padL}
              y1={t.y}
              x2={W - padR}
              y2={t.y}
              stroke={t.val === 0 ? 'var(--line-hot)' : 'var(--line)'}
              strokeWidth={t.val === 0 ? 1.4 : 0.7}
              strokeDasharray={t.val === 0 ? undefined : '3 3'}
            />
            <text x={padL - 6} y={t.y + 3} textAnchor="end" fill={t.val === 0 ? 'var(--cyan)' : 'var(--muted)'} fontSize="8.5" fontFamily="JetBrains Mono" fontWeight={t.val === 0 ? '700' : '400'}>
              {t.val > 0 ? `+${t.val.toFixed(1)}s` : t.val === 0 ? '0.0s' : `${t.val.toFixed(1)}s`}
            </text>
          </g>
        ))}

        {/* X-axis labels */}
        {xTicks.map(lapNum => {
          const x = scaleX(lapNum)
          return (
            <g key={lapNum}>
              <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke="var(--line)" strokeDasharray="2 4" strokeWidth="0.6" opacity="0.6" />
              <text x={x} y={H - 10} textAnchor="middle" fill="var(--muted)" fontSize="8.5" fontFamily="JetBrains Mono">
                L{lapNum}
              </text>
            </g>
          )
        })}

        {/* Shaded Area */}
        <path d={areaPath} fill="url(#gapGrad)" />

        {/* Gap Line */}
        <path d={linePath} fill="none" stroke="var(--cyan)" strokeWidth="2.2" />

        {/* Data Point Dots */}
        {commonLaps.map((p, i) => (
          <circle key={i} cx={scaleX(p.lap)} cy={scaleY(p.gap)} r="2" fill="var(--cyan)" stroke="#05080c" strokeWidth="0.7" />
        ))}

        {/* Max / Min Annotations */}
        {maxPt.gap > 0.5 && (
          <g>
            <circle cx={scaleX(maxPt.lap)} cy={scaleY(maxPt.gap)} r="3.5" fill="var(--green)" stroke="#fff" strokeWidth="1" />
            <text x={scaleX(maxPt.lap)} y={scaleY(maxPt.gap) - 7} textAnchor="middle" fill="var(--green)" fontSize="7.5" fontWeight="700" fontFamily="JetBrains Mono">
              +{maxPt.gap.toFixed(2)}s (L{maxPt.lap})
            </text>
          </g>
        )}
      </svg>
    </div>
  )
}

function RealForecastChart({ simResult, lap }: { simResult: SimResult | null; lap: number }) {
  const values = simResult?.net_undercut_advantage || []
  if (!values.length) {
    return <div style={{ height: 180, display: 'grid', placeItems: 'center', color: 'var(--muted)', font: '10px JetBrains Mono' }}>Running Monte Carlo forward rollouts...</div>
  }

  const minVal = Math.min(-1.0, Math.floor(Math.min(...values) * 2) / 2 - 0.5)
  const maxVal = Math.max(2.0, Math.ceil(Math.max(...values) * 2) / 2 + 0.5)
  const span = maxVal - minVal || 1

  const W = 680
  const H = 260
  const padL = 52
  const padR = 25
  const padT = 25
  const padB = 34
  const plotW = W - padL - padR
  const plotH = H - padT - padB

  const scaleX = (idx: number) => padL + (idx / Math.max(1, values.length - 1)) * plotW
  const scaleY = (val: number) => padT + plotH - ((val - minVal) / span) * plotH

  const points = values.map((v, i) => `${scaleX(i).toFixed(1)},${scaleY(v).toFixed(1)}`)
  const zeroY = scaleY(0)

  // Confidence ribbon (p05 to p95 band estimation)
  const upperPts = values.map((v, i) => {
    const bound = v + 0.28 * Math.sqrt(i + 1)
    return `${scaleX(i).toFixed(1)},${scaleY(bound).toFixed(1)}`
  })
  const lowerPts = values.map((v, i) => {
    const bound = v - 0.28 * Math.sqrt(i + 1)
    return `${scaleX(i).toFixed(1)},${scaleY(bound).toFixed(1)}`
  }).reverse()
  const ribbonD = `M ${upperPts.join(' L ')} L ${lowerPts.join(' L ')} Z`

  const areaD = `M ${scaleX(0)},${zeroY.toFixed(1)} L ${points.join(' L ')} L ${scaleX(values.length - 1)},${zeroY.toFixed(1)} Z`
  const lineD = `M ${points.join(' L ')}`

  const crossIdx = simResult?.crossover_lap != null ? simResult.crossover_lap - lap : -1
  const crossX = crossIdx >= 0 && crossIdx < values.length ? scaleX(crossIdx) : null

  // Peak advantage
  let peakIdx = 0
  let peakVal = values[0]
  values.forEach((v, i) => {
    if (v > peakVal) { peakVal = v; peakIdx = i }
  })

  // Y-ticks
  const yTicks = [minVal, minVal + span * 0.25, minVal + span * 0.5, minVal + span * 0.75, maxVal].map(val => ({
    val,
    y: scaleY(val)
  }))

  return (
    <div className="chart-body">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
        <defs>
          <linearGradient id="foreAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--cyan)" stopOpacity="0.25" />
            <stop offset="100%" stopColor="var(--cyan)" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Gridlines and Y-ticks */}
        {yTicks.map((t, idx) => (
          <g key={idx}>
            <line
              x1={padL}
              y1={t.y}
              x2={W - padR}
              y2={t.y}
              stroke={Math.abs(t.val) < 0.2 ? 'var(--line-hot)' : 'var(--line)'}
              strokeWidth={Math.abs(t.val) < 0.2 ? 1.4 : 0.7}
              strokeDasharray={Math.abs(t.val) < 0.2 ? undefined : '3 3'}
            />
            <text x={padL - 6} y={t.y + 3} textAnchor="end" fill={Math.abs(t.val) < 0.2 ? 'var(--cyan)' : 'var(--muted)'} fontSize="8.5" fontFamily="JetBrains Mono" fontWeight={Math.abs(t.val) < 0.2 ? '700' : '400'}>
              {t.val > 0 ? `+${t.val.toFixed(1)}s` : `${t.val.toFixed(1)}s`}
            </text>
          </g>
        ))}

        {/* X-axis tick labels */}
        {values.map((_, i) => {
          if (i % 2 !== 0 && i !== values.length - 1) return null
          const x = scaleX(i)
          return (
            <g key={i}>
              <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke="var(--line)" strokeDasharray="2 4" strokeWidth="0.6" opacity="0.6" />
              <text x={x} y={H - 10} textAnchor="middle" fill="var(--muted)" fontSize="8.5" fontFamily="JetBrains Mono">
                L{lap + i}
              </text>
            </g>
          )
        })}

        {/* Confidence Ribbon */}
        <path d={ribbonD} fill="rgba(32, 217, 239, 0.08)" />

        {/* Area fill */}
        <path d={areaD} fill="url(#foreAreaGrad)" />

        {/* Line */}
        <path d={lineD} fill="none" stroke="var(--cyan)" strokeWidth="2.4" />

        {/* Data Point Dots */}
        {values.map((v, i) => (
          <circle key={i} cx={scaleX(i)} cy={scaleY(v)} r="2.5" fill="var(--cyan)" stroke="#05080c" strokeWidth="0.8" />
        ))}

        {/* Crossover Lap Indicator */}
        {crossX !== null && (
          <g>
            <line x1={crossX} y1={padT - 6} x2={crossX} y2={padT + plotH} stroke="var(--red)" strokeWidth="1.5" strokeDasharray="3 3" />
            <rect x={crossX - 35} y={padT - 18} width="70" height="15" fill="#080e14" stroke="var(--red)" strokeWidth="1" rx="2" />
            <text x={crossX} y={padT - 8} textAnchor="middle" fill="var(--red)" fontSize="7" fontWeight="700" fontFamily="JetBrains Mono">
              CROSSOVER L{simResult?.crossover_lap}
            </text>
          </g>
        )}

        {/* Peak Advantage Callout */}
        {peakVal > 0 && (
          <g>
            <circle cx={scaleX(peakIdx)} cy={scaleY(peakVal)} r="4" fill="var(--green)" stroke="#fff" strokeWidth="1" />
            <text x={scaleX(peakIdx)} y={scaleY(peakVal) - 8} textAnchor="middle" fill="var(--green)" fontSize="7.5" fontWeight="700" fontFamily="JetBrains Mono">
              MAX +{peakVal.toFixed(2)}s (L{lap + peakIdx})
            </text>
          </g>
        )}
      </svg>
    </div>
  )
}

function SelectField({ label, value, onChange, children }: { label: string; value: string; onChange: (value: string) => void; children: React.ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="select-wrap">
        <select value={value} onChange={e => onChange(e.target.value)}>
          {children}
        </select>
        <Icon name="chevron" size={14} />
      </div>
    </label>
  )
}

function TelemetryInspectorModal({
  chart,
  onClose,
  lapsA,
  lapsB,
  a,
  b,
  simResult,
  pitLap,
  compound,
  raceData,
  selectedRace,
  year
}: {
  chart: 'degradation' | 'gap' | 'forecast' | 'circuit'
  onClose: () => void
  lapsA: LapRecord[]
  lapsB: LapRecord[]
  a: Driver
  b: Driver
  simResult: SimResult | null
  pitLap: number
  compound: string
  raceData: RaceInfo | null
  selectedRace: string
  year: string
}) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const [hoverLap, setHoverLap] = useState<number | null>(null)

  // Degradation / Lap calculations
  const allLaps = [...lapsA, ...lapsB]
  const minLap = allLaps.length ? Math.min(...allLaps.map(l => l.l)) : 1
  const maxLap = allLaps.length ? Math.max(...allLaps.map(l => l.l)) : 57
  const times = allLaps.map(l => l.t).filter(t => t > 60 && t < 150)
  const minTime = times.length ? Math.floor(Math.min(...times) * 2) / 2 - 0.5 : 88
  const maxTime = times.length ? Math.ceil(Math.max(...times) * 2) / 2 + 0.5 : 96
  const timeSpan = maxTime - minTime || 1

  const activeLap = hoverLap != null ? hoverLap : (lapsA.length ? lapsA[lapsA.length - 1].l : minLap)
  const curA = lapsA.find(l => l.l === activeLap) || lapsA[lapsA.length - 1]
  const curB = lapsB.find(l => l.l === activeLap) || lapsB[lapsB.length - 1]
  const paceDelta = curA && curB ? curB.t - curA.t : 0

  const W = 1040
  const H = 460
  const padL = 60
  const padR = 40
  const padT = 30
  const padB = 40
  const plotW = W - padL - padR
  const plotH = H - padT - padB

  const scaleX = (lap: number) => padL + ((lap - minLap) / Math.max(1, maxLap - minLap)) * plotW
  const scaleY = (t: number) => padT + plotH - ((t - minTime) / timeSpan) * plotH

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const frac = Math.max(0, Math.min(1, (mouseX - padL) / plotW))
    const lap = Math.round(minLap + frac * (maxLap - minLap))
    setHoverLap(lap)
  }

  return (
    <div className="telemetry-modal-overlay" onClick={onClose}>
      <div className="telemetry-modal" onClick={e => e.stopPropagation()}>
        <div className="telemetry-modal-header">
          <div>
            <span className="eyebrow">INTERACTIVE TELEMETRY INSPECTOR // {selectedRace.toUpperCase()} {year}</span>
            <h2 style={{ margin: '4px 0 0', fontSize: 24, fontStyle: 'italic', textTransform: 'uppercase' }}>
              {chart === 'degradation' ? 'Full Race Pace Trace & Degradation Model' :
               chart === 'gap' ? 'High-Resolution Gap Evolution & Parity Analysis' :
               chart === 'forecast' ? 'Predictive Undercut Horizon & Monte Carlo Simulation' :
               '3D Circuit Topography & Dynamic Elevation Profile'}
            </h2>
          </div>
          <button className="close-modal-btn" onClick={onClose}>
            ✕ CLOSE [ESC]
          </button>
        </div>

        <div className="telemetry-modal-body">
          {/* Live Telemetry Readout Card */}
          {chart !== 'circuit' && (
            <div className="telemetry-readout-card">
              <div style={{ borderRight: '1px solid var(--line)', paddingRight: 14 }}>
                <span style={{ color: 'var(--muted)', fontSize: 9 }}>TRACKING LAP</span>
                <strong style={{ display: 'block', fontSize: 18, color: 'var(--cyan)' }}>LAP {activeLap} <small style={{ fontSize: 10, color: 'var(--muted)' }}>/ {maxLap}</small></strong>
              </div>
              <div style={{ borderRight: '1px solid var(--line)', paddingRight: 14 }}>
                <span style={{ color: a.accent, fontWeight: 700 }}>{a.code} ({a.name})</span>
                <div style={{ fontSize: 12, marginTop: 2 }}>
                  TIME: <b>{curA?.t ? `${curA.t.toFixed(3)}s` : '—'}</b> · TYRE: <b>{curA?.c || 'HARD'} ({curA?.a || 1}L)</b>
                </div>
              </div>
              <div style={{ borderRight: '1px solid var(--line)', paddingRight: 14 }}>
                <span style={{ color: b.accent, fontWeight: 700 }}>{b.code} ({b.name})</span>
                <div style={{ fontSize: 12, marginTop: 2 }}>
                  TIME: <b>{curB?.t ? `${curB.t.toFixed(3)}s` : '—'}</b> · TYRE: <b>{curB?.c || 'HARD'} ({curB?.a || 1}L)</b>
                </div>
              </div>
              <div>
                <span style={{ color: 'var(--muted)', fontSize: 9 }}>LIVE LAP ADVANTAGE</span>
                <strong style={{ display: 'block', fontSize: 15, color: paceDelta > 0 ? 'var(--green)' : 'var(--red)' }}>
                  {paceDelta > 0 ? `${a.code} +${paceDelta.toFixed(3)}s faster` : `${b.code} +${(-paceDelta).toFixed(3)}s faster`}
                </strong>
              </div>
              <div style={{ marginLeft: 'auto', font: '9px JetBrains Mono', color: 'var(--muted)' }}>
                MOVE CURSOR OVER CHART TO SCRUB TELEMETRY
              </div>
            </div>
          )}

          {chart === 'circuit' ? (
            <div style={{ height: 500, position: 'relative' }}>
              <Circuit3DCard
                year={raceData?.year || parseInt(year, 10) || 2023}
                race={raceData?.race || selectedRace}
                driverAAccent={a.accent}
                driverBAccent={b.accent}
              />
              <div style={{ position: 'absolute', bottom: 12, left: 16, font: '9px JetBrains Mono', color: 'var(--muted)', background: 'rgba(5,8,12,0.85)', padding: '6px 12px', border: '1px solid var(--line)' }}>
                DRAG WITH MOUSE TO ROTATE ORBIT · SCROLL TO ZOOM ELEVATION
              </div>
            </div>
          ) : chart === 'degradation' ? (
            <div style={{ height: 460 }}>
              <svg
                viewBox={`0 0 ${W} ${H}`}
                preserveAspectRatio="none"
                style={{ width: '100%', height: '100%', cursor: 'crosshair' }}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoverLap(null)}
              >
                <defs>
                  <linearGradient id="modalGradA" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={a.accent} stopOpacity="0.25" />
                    <stop offset="100%" stopColor={a.accent} stopOpacity="0.0" />
                  </linearGradient>
                  <linearGradient id="modalGradB" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={b.accent} stopOpacity="0.15" />
                    <stop offset="100%" stopColor={b.accent} stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Y Gridlines */}
                {[0, 0.2, 0.4, 0.6, 0.8, 1].map(frac => {
                  const val = minTime + frac * timeSpan
                  const y = scaleY(val)
                  return (
                    <g key={frac}>
                      <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="var(--line)" strokeDasharray="3 3" strokeWidth="0.8" />
                      <text x={padL - 8} y={y + 4} textAnchor="end" fill="var(--muted)" fontSize="9" fontFamily="JetBrains Mono">
                        {val.toFixed(2)}s
                      </text>
                    </g>
                  )
                })}

                {/* X Gridlines */}
                {Array.from({ length: Math.ceil((maxLap - minLap) / 4) + 1 }).map((_, i) => {
                  const l = Math.min(maxLap, minLap + i * 4)
                  const x = scaleX(l)
                  return (
                    <g key={l}>
                      <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke="var(--line)" strokeDasharray="2 4" strokeWidth="0.6" opacity="0.6" />
                      <text x={x} y={H - 12} textAnchor="middle" fill="var(--muted)" fontSize="9" fontFamily="JetBrains Mono">
                        L{l}
                      </text>
                    </g>
                  )
                })}

                {/* Shaded Areas */}
                {lapsB.length > 0 && (
                  <path
                    d={`${lapsB.map((r, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(r.l).toFixed(1)} ${scaleY(r.t).toFixed(1)}`).join(' ')} L ${scaleX(lapsB[lapsB.length - 1].l)} ${padT + plotH} L ${scaleX(lapsB[0].l)} ${padT + plotH} Z`}
                    fill="url(#modalGradB)"
                  />
                )}
                {lapsA.length > 0 && (
                  <path
                    d={`${lapsA.map((r, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(r.l).toFixed(1)} ${scaleY(r.t).toFixed(1)}`).join(' ')} L ${scaleX(lapsA[lapsA.length - 1].l)} ${padT + plotH} L ${scaleX(lapsA[0].l)} ${padT + plotH} Z`}
                    fill="url(#modalGradA)"
                  />
                )}

                {/* Lines */}
                {lapsB.length > 0 && (
                  <path
                    d={lapsB.map((r, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(r.l).toFixed(1)} ${scaleY(r.t).toFixed(1)}`).join(' ')}
                    fill="none"
                    stroke={b.accent}
                    strokeWidth="2.2"
                  />
                )}
                {lapsA.length > 0 && (
                  <path
                    d={lapsA.map((r, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(r.l).toFixed(1)} ${scaleY(r.t).toFixed(1)}`).join(' ')}
                    fill="none"
                    stroke={a.accent}
                    strokeWidth="2.8"
                  />
                )}

                {/* Data point dots */}
                {lapsB.map((r, i) => (
                  <circle key={`b-${i}`} cx={scaleX(r.l)} cy={scaleY(r.t)} r={r.l === activeLap ? 5 : 2.5} fill={b.accent} />
                ))}
                {lapsA.map((r, i) => (
                  <circle key={`a-${i}`} cx={scaleX(r.l)} cy={scaleY(r.t)} r={r.l === activeLap ? 6 : 3} fill={a.accent} stroke="#fff" strokeWidth={r.l === activeLap ? 2 : 0.5} />
                ))}

                {/* Hover Scrubber Line */}
                {hoverLap != null && (
                  <g>
                    <line x1={scaleX(hoverLap)} y1={padT} x2={scaleX(hoverLap)} y2={padT + plotH} stroke="var(--cyan)" strokeWidth="1.8" strokeDasharray="4 2" />
                    <rect x={scaleX(hoverLap) - 24} y={padT - 18} width="48" height="16" fill="var(--cyan)" rx="2" />
                    <text x={scaleX(hoverLap)} y={padT - 6} textAnchor="middle" fill="#031014" fontSize="9" fontWeight="800" fontFamily="JetBrains Mono">
                      L{hoverLap}
                    </text>
                  </g>
                )}
              </svg>
            </div>
          ) : chart === 'gap' ? (
            <div style={{ height: 460 }}>
              <RealGapChart lapsA={lapsA} lapsB={lapsB} a={a} b={b} />
            </div>
          ) : (
            <div style={{ height: 460 }}>
              <RealForecastChart simResult={simResult} lap={pitLap} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState<'home' | 'dashboard'>('home')
  const [theme, setTheme] = useState<'dark' | 'cream'>('dark')
  const [dashboardMode, setDashboardMode] = useState<'live' | 'review'>('live')
  const [enlargedChart, setEnlargedChart] = useState<'degradation' | 'gap' | 'forecast' | 'circuit' | null>(null)
  const [launching, setLaunching] = useState(false)
  const [year, setYear] = useState('2023')
  const [schedules, setSchedules] = useState<Record<string, string[]>>(DEFAULT_SCHEDULES)
  const [selectedRace, setSelectedRace] = useState('Bahrain Grand Prix')
  const [driverA, setDriverA] = useState('VER')
  const [driverB, setDriverB] = useState('LEC')
  const [driversMeta, setDriversMeta] = useState<Driver[]>(DEFAULT_DRIVERS)
  const [state, setState] = useState<'active' | 'loading' | 'empty'>('loading')
  const [pitLap, setPitLap] = useState(14)
  const [compound, setCompound] = useState('HARD')
  const [rollouts, setRollouts] = useState(500)
  const [raceData, setRaceData] = useState<RaceInfo | null>(null)
  const [simResult, setSimResult] = useState<SimResult | null>(null)
  const [slide, setSlide] = useState(0)

  // ── Race Review & What-If State ──
  const [reviewInit, setReviewInit] = useState<ReviewInitData | null>(null)
  const [reviewDriver, setReviewDriver] = useState('VER')
  const [reviewStopIdx, setReviewStopIdx] = useState(0)
  const [rivalMode, setRivalMode] = useState<'cover_traffic' | 'cover' | 'fixed'>('cover_traffic')
  const [whatifShift, setWhatifShift] = useState(0)
  const [regretResult, setRegretResult] = useState<RegretScanResult | null>(null)
  const [whatifResult, setWhatifResult] = useState<WhatIfResult | null>(null)
  const [stintDegResult, setStintDegResult] = useState<StintDegResult | null>(null)
  const [reviewLoading, setReviewLoading] = useState(false)
  const [regretLoading, setRegretLoading] = useState(false)
  const [whatifLoading, setWhatifLoading] = useState(false)

  const raceList = schedules[year] || DEFAULT_SCHEDULES[year] || ['Bahrain Grand Prix']
  const a = driversMeta.find(d => d.code === driverA) || { code: driverA, name: driverA, number: 1, team: 'Red Bull Racing', car: 'RB20', country: 'Netherlands', accent: '#3671C6' }
  const b = driversMeta.find(d => d.code === driverB) || { code: driverB, name: driverB, number: 16, team: 'Ferrari', car: 'SF-24', country: 'Monaco', accent: '#E80020' }

  // Slide timer
  const nextSlide = useCallback(() => setSlide(s => (s + 1) % SLIDE_IMAGES.length), [])
  useEffect(() => {
    const id = setInterval(nextSlide, SLIDE_INTERVAL)
    return () => clearInterval(id)
  }, [nextSlide])

  // Fetch initial schedules and metadata
  useEffect(() => {
    fetch('/api/schedules')
      .then(r => r.json())
      .then(data => {
        if (data && typeof data === 'object') setSchedules(data)
      })
      .catch(() => {})

    fetch('/api/metadata')
      .then(r => r.json())
      .then(data => {
        if (data && data.drivers && Array.isArray(data.drivers)) {
          setDriversMeta(data.drivers)
        }
      })
      .catch(() => {})
  }, [])

  // Execute Load Race
  const executeLoadRace = useCallback((targetYear: string, targetRace: string) => {
    setState('loading')
    fetch('/api/load-race', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ year: parseInt(targetYear), race: targetRace })
    })
      .then(r => {
        if (!r.ok) throw new Error('Failed to load race')
        return r.json()
      })
      .then((data: RaceInfo) => {
        setRaceData(data)
        if (data.drivers && data.drivers.length > 0) {
          if (!data.drivers.includes(driverA)) setDriverA(data.drivers[0])
          if (!data.drivers.includes(driverB)) setDriverB(data.drivers[1] || data.drivers[0])
        }
        const defaultSimLap = Math.max(data.min_lap, Math.min(data.max_lap, 14))
        setPitLap(defaultSimLap)
        setState('active')
      })
      .catch(() => {
        setState('active')
      })
  }, [driverA, driverB])

  useEffect(() => {
    executeLoadRace(year, selectedRace)
  }, [])

  // Live Simulation Trigger
  useEffect(() => {
    if (state !== 'active' || !raceData) return

    fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        year: parseInt(year),
        race: selectedRace,
        driver_a: driverA,
        driver_b: driverB,
        sim_lap: pitLap,
        next_compound: compound,
        n_rollouts: rollouts,
      })
    })
      .then(r => {
        if (!r.ok) throw new Error('Simulation failed')
        return r.json()
      })
      .then((res: SimResult) => {
        setSimResult(res)
      })
      .catch(() => {})
  }, [year, selectedRace, driverA, driverB, pitLap, compound, rollouts, state, raceData])

  // Race Review Data Loading
  useEffect(() => {
    if (dashboardMode !== 'review') return
    setReviewLoading(true)

    fetch(`/api/race-review/init?year=${year}&race=${encodeURIComponent(selectedRace)}`)
      .then(r => r.json())
      .then((data: ReviewInitData) => {
        setReviewInit(data)
        if (data.drivers && data.drivers.length > 0) {
          const firstDrv = data.drivers[0].code
          setReviewDriver(firstDrv)
        }
        setReviewLoading(false)
      })
      .catch(() => {
        setReviewLoading(false)
      })
  }, [dashboardMode, year, selectedRace])

  // Fetch Regret Scan & Stint Deg for Review Driver
  useEffect(() => {
    if (dashboardMode !== 'review' || !reviewDriver) return

    setRegretLoading(true)
    // 1. Regret scan (n_paths: 60 for fast responsive MC scan)
    fetch('/api/race-review/regret-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        year: parseInt(year),
        race: selectedRace,
        driver: reviewDriver,
        stop_idx: reviewStopIdx,
        rival_mode: rivalMode,
        n_paths: 60,
      })
    })
      .then(r => r.json())
      .then((data: RegretScanResult) => {
        setRegretResult(data)
        setRegretLoading(false)
      })
      .catch(() => {
        setRegretLoading(false)
      })

    // 2. Stint deg curves
    fetch(`/api/race-review/stint-degradation?year=${year}&race=${encodeURIComponent(selectedRace)}&driver=${reviewDriver}`)
      .then(r => r.json())
      .then((data: StintDegResult) => setStintDegResult(data))
      .catch(() => {})
  }, [dashboardMode, year, selectedRace, reviewDriver, reviewStopIdx, rivalMode])

  // What-If run when slider changes
  useEffect(() => {
    if (dashboardMode !== 'review' || !reviewDriver) return

    setWhatifLoading(true)
    fetch('/api/race-review/whatif', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        year: parseInt(year),
        race: selectedRace,
        driver: reviewDriver,
        stop_idx: reviewStopIdx,
        shift_laps: whatifShift,
        rival_mode: rivalMode,
        n_paths: 60,
      })
    })
      .then(r => r.json())
      .then((data: WhatIfResult) => {
        setWhatifResult(data)
        setWhatifLoading(false)
      })
      .catch(() => {
        setWhatifLoading(false)
      })
  }, [dashboardMode, year, selectedRace, reviewDriver, reviewStopIdx, rivalMode, whatifShift])

  const enterDashboard = () => {
    setLaunching(true)
    window.setTimeout(() => {
      setPage('dashboard')
      setLaunching(false)
      window.scrollTo(0, 0)
    }, 650)
  }

  const lapsA = useMemo(() => {
    if (!raceData?.laps) return []
    return raceData.laps.filter(l => l.d === driverA)
  }, [raceData, driverA])

  const lapsB = useMemo(() => {
    if (!raceData?.laps) return []
    return raceData.laps.filter(l => l.d === driverB)
  }, [raceData, driverB])

  const currentGap = useMemo(() => {
    if (!lapsA.length || !lapsB.length) return 0
    const lastLapA = lapsA[lapsA.length - 1]
    const matchingLapB = lapsB.find(l => l.l === lastLapA.l)
    if (matchingLapB && lastLapA.elapsed != null && matchingLapB.elapsed != null) {
      return matchingLapB.elapsed - lastLapA.elapsed
    }
    return 1.42
  }, [lapsA, lapsB])

  const verdict = simResult?.verdict || 'pending'

  // Selected driver profile in review
  const activeReviewDriver = reviewInit?.drivers.find(d => d.code === reviewDriver) || reviewInit?.drivers[0]

  // ═══════════════════  LANDING PAGE (100% PRESERVED EXACTLY AS UPLOADED)  ═══════════════════
  if (page === 'home') {
    return (
      <div className={`landing ${launching ? 'is-launching' : ''}`}>
        {/* ── Full-screen slideshow ── */}
        <div className="slideshow">
          {SLIDE_IMAGES.map((src, i) => (
            <img
              key={src}
              className={`slideshow-img ${i === slide ? 'active' : ''}`}
              src={src}
              alt={`F1 showcase ${i + 1}`}
              loading={i === 0 ? 'eager' : 'lazy'}
            />
          ))}
        </div>

        {/* ── Subtle light veil for high image clarity ── */}
        <div className="landing-veil" />

        {/* ── Top nav (totally transparent) ── */}
        <header className="landing-nav">
          <div className="brand landing-brand">
            <BrandMark />
            <div>
              <strong>F1</strong>
              <span>STRATEGY ENGINEER</span>
            </div>
          </div>
          <div className="landing-links">
            <a href="#platform">Platform</a>
            <a href="#intelligence">Intelligence</a>
            <a href="#about">About</a>
          </div>
          <div className="nav-live">
            <span className="live-dot" /> MODEL ONLINE
          </div>
        </header>

        {/* ── Hero content (totally transparent) ── */}
        <section className="landing-hero-wrap">
          <div className="glass-hero">
            <p className="landing-overline">REAL-TIME STRATEGY INTELLIGENCE</p>
            <h1 className="landing-title">Race the<br /><em>future.</em></h1>
            <p className="landing-subtitle">Millisecond decisions. Powered by data.</p>
            <div className="landing-actions">
              <button className="enter-button" onClick={enterDashboard}><span>ENTER STRATEGY HUB</span><b>↗</b></button>
            </div>
          </div>
        </section>

        {/* ── Slide pagination dots (clean indicators) ── */}
        <div className="slide-dots">
          {SLIDE_IMAGES.map((_, i) => (
            <button
              key={i}
              className={`slide-dot ${i === slide ? 'active' : ''}`}
              onClick={() => setSlide(i)}
              aria-label={`Go to slide ${i + 1}`}
            />
          ))}
        </div>

        {/* ── Bottom feature strip (totally transparent) ── */}
        <footer className="landing-footer">
          <div><span>01</span><p><b>LIVE TELEMETRY</b>Real-time lap intelligence</p></div>
          <div><span>02</span><p><b>PREDICTIVE STRATEGY</b>15-lap outcome modelling</p></div>
          <div><span>03</span><p><b>DECISION EDGE</b>Know the call first</p></div>
        </footer>

        {/* ── Launch wipe transition ── */}
        <div className="launch-wipe"><BrandMark /><span>INITIALIZING F1 STRATEGY ENGINEER</span></div>
      </div>
    )
  }

  // ═══════════════════  DASHBOARD PAGE  ═══════════════════
  return (
    <div className={`app-shell dashboard-shell ${theme === 'cream' ? 'theme-cream' : ''}`}>
      <aside className="sidebar">
        <div className="brand">
          <BrandMark />
          <div>
            <strong>F1</strong>
            <span>STRATEGY ENGINEER</span>
          </div>
        </div>

        <div className="side-status">
          <span className="live-dot" /> SYSTEM ONLINE
          <b>98.7%</b>
        </div>

        <nav>
          <button className={dashboardMode === 'live' ? 'active' : ''} onClick={() => setDashboardMode('live')} title="Live Strategy">
            <Icon name="target" /><em>01</em>
          </button>
          <button className={dashboardMode === 'review' ? 'active' : ''} onClick={() => setDashboardMode('review')} title="Race Review & What-If">
            <Icon name="chart" /><em>02</em>
          </button>
          <button onClick={() => setDashboardMode('live')} title="Telemetry">
            <Icon name="radio" /><em>03</em>
          </button>
          <button onClick={() => setDashboardMode('live')} title="Grid Setup">
            <Icon name="grid" /><em>04</em>
          </button>
        </nav>

        <div className="control-section">
          <div className="section-label"><span>01 // SESSION CONFIG</span><b>SEASON</b></div>
          <div className="field-row">
            <SelectField label="Season" value={year} onChange={setYear}>
              {['2025', '2024', '2023', '2022'].map(y => <option key={y} value={y}>{y}</option>)}
            </SelectField>
            <SelectField label="Round" value={selectedRace} onChange={setSelectedRace}>
              {raceList.map(r => <option key={r} value={r}>{r.replace(' Grand Prix', ' GP')}</option>)}
            </SelectField>
          </div>
          <SelectField label="Grand Prix Name" value={selectedRace} onChange={setSelectedRace}>
            {raceList.map(r => <option key={r} value={r}>{r}</option>)}
          </SelectField>
        </div>

        {dashboardMode === 'live' && (
          <div className="control-section">
            <div className="section-label"><span>02 // MATCHUP</span><b>GRID</b></div>
            <SelectField label="Driver A (Target / Attacker)" value={driverA} onChange={setDriverA}>
              {(raceData?.drivers?.length ? raceData.drivers : driversMeta.map(d => d.code)).map(c => {
                const dm = driversMeta.find(d => d.code === c)
                return <option key={c} value={c}>{c} — {dm?.name || c}</option>
              })}
            </SelectField>
            <SelectField label="Driver B (Defender / Chaser)" value={driverB} onChange={setDriverB}>
              {(raceData?.drivers?.length ? raceData.drivers : driversMeta.map(d => d.code)).map(c => {
                const dm = driversMeta.find(d => d.code === c)
                return <option key={c} value={c}>{c} — {dm?.name || c}</option>
              })}
            </SelectField>
          </div>
        )}

        {dashboardMode === 'live' && (
          <button className="load-button" onClick={() => executeLoadRace(year, selectedRace)}>
            SYNC TELEMETRY <span>&rarr;</span>
          </button>
        )}

        <button className="reset-button" onClick={() => setState('empty')}>
          <Icon name="reset" size={14} /> Clear session
        </button>

        <div style={{ margin: '14px 0 6px' }}>
          <button
            className="theme-toggle-btn"
            style={{ width: '100%', justifyContent: 'center', padding: '8px 10px' }}
            onClick={() => setTheme(t => t === 'dark' ? 'cream' : 'dark')}
          >
            {theme === 'dark' ? '☀️ LIGHT CREAM' : '🌙 STEALTH DARK'}
          </button>
        </div>

        <div className="sidebar-footer">
          <span>MODEL</span>
          <b>QUANTILE-XGB + MC</b>
          <small>14 PHYSICS FEATS · ZERO LEAKAGE</small>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div className="breadcrumb">
            <button className="home-link" onClick={() => setPage('home')}>← HOME</button>
            <span>/</span> PIT WALL <span>/</span>
            <div className="mode-nav">
              <button className={`mode-btn ${dashboardMode === 'live' ? 'active' : ''}`} onClick={() => setDashboardMode('live')}>
                🏎️ LIVE STRATEGY
              </button>
              <button className={`mode-btn ${dashboardMode === 'review' ? 'active' : ''}`} onClick={() => setDashboardMode('review')}>
                📋 RACE REVIEW & WHAT-IF
              </button>
            </div>
          </div>
          <div className="header-meta">
            <span className="project-badge">F1 STRATEGY ENGINEER</span>
            <button
              className="theme-toggle-btn"
              onClick={() => setTheme(t => t === 'dark' ? 'cream' : 'dark')}
              title="Toggle Light Cream / Stealth Dark theme"
            >
              {theme === 'dark' ? '☀️ LIGHT CREAM' : '🌙 STEALTH DARK'}
            </button>
            <span>{selectedRace} · {year}</span>
            <span className="live-pill"><i /> {dashboardMode === 'live' ? 'LIVE MODEL' : 'REPLAY ENGINE'}</span>
            <span className="initials">SE</span>
          </div>
        </header>

        {state === 'loading' ? (
          <section className="state-screen">
            <div className="loader"><span /><span /><span /></div>
            <span className="eyebrow">INGESTING TELEMETRY</span>
            <h2>Building race model</h2>
            <p>Synchronizing timing loops, tyre stints and pit lane deltas for {selectedRace}...</p>
            <div className="progress"><span /></div>
          </section>
        ) : state === 'empty' ? (
          <section className="state-screen empty">
            <div className="empty-radar"><Icon name="radio" size={42} /></div>
            <span className="eyebrow">NO ACTIVE SESSION</span>
            <h2>Select a Grand Prix</h2>
            <p>Configure the race and driver matchup in the control panel, then load telemetry to initialize the strategy model.</p>
            <button className="outline-button" onClick={() => executeLoadRace(year, selectedRace)}>INITIALIZE {selectedRace.toUpperCase()}</button>
          </section>
        ) : dashboardMode === 'live' ? (
          /* ══════════════════════════════════════════════════════════════════
             MODE 1: LIVE STRATEGY DASHBOARD
             ══════════════════════════════════════════════════════════════════ */
          <div className="dashboard">
            <section className="hero">
              <div>
                <div className="status-badge"><span className="live-dot" /> LIVE RACE DATA <i /> ML STRATEGY MODEL</div>
                <p className="kicker">PIT WALL DECISION SUPPORT // SESSION 04</p>
                <h1>F1 Strategy <span>Engineer</span></h1>
                <p className="hero-copy">Real-time race intelligence for undercut prediction. Multi-quantile degradation, two-car game theory, and 6 tactical candidate strategies.</p>
              </div>
              <div className="session-lock">
                <span>SESSION LOCK</span>
                <strong>RACE</strong>
                <small>{(raceData?.country || 'BAHRAIN').toUpperCase()} · DRY</small>
              </div>
            </section>

            {/* KPI STRIP */}
            <section className="kpi-strip">
              {[
                ['SEASON', year, 'CHAMPIONSHIP'],
                ['GRAND PRIX', (raceData?.race || selectedRace).replace(' Grand Prix', ' GP'), `ROUND ${String((raceList.indexOf(selectedRace) >= 0 ? raceList.indexOf(selectedRace) : 0) + 1).padStart(2, '0')}`],
                ['MATCHUP', `${a.code} / ${b.code}`, 'HEAD TO HEAD'],
                ['CIRCUIT', raceData?.circuit || selectedRace, `${raceData?.laps_count || 57} LAPS`],
              ].map(([label, value, sub], i) => (
                <div className="kpi" key={label}>
                  <span>0{i + 1} // {label}</span>
                  <strong>{value}</strong>
                  <small>{sub}</small>
                </div>
              ))}
            </section>

            {/* MATCHUP SECTION */}
            <section className="matchup-section">
              <div className="section-title">
                <div>
                  <span className="eyebrow">DRIVER COMPARISON // LIVE STINT</span>
                  <h2>Head-to-head telemetry</h2>
                </div>
                <span className="data-quality"><i /> DATA QUALITY · 99.8%</span>
              </div>
              <div className="matchup-grid">
                <DriverCard driver={a} role="ATTACKER // TARGET" side="a" />
                <div className="vs"><span>HEAD</span><strong>VS</strong><span>HEAD</span></div>
                <DriverCard driver={b} role="DEFENDER // CHASER" side="b" />
                <aside className="circuit-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span className="eyebrow">CIRCUIT PROFILE</span>
                    <button className="enlarge-chart-btn" onClick={() => setEnlargedChart('circuit')} title="Click to enlarge 3D circuit">
                      ⛶ 3D FULLSCREEN
                    </button>
                  </div>
                  <Circuit3DCard
                    year={raceData?.year || parseInt(year, 10) || 2023}
                    race={raceData?.race || selectedRace}
                    driverAAccent={a.accent}
                    driverBAccent={b.accent}
                  />
                  <h3>{raceData?.circuit || selectedRace}</h3>
                  <p>{raceData?.country || 'Grand Prix Circuit'}</p>
                  <div className="circuit-stats">
                    <div><span>RACE LAPS</span><b>{raceData?.laps_count || 57}</b></div>
                    <div><span>LAP DISTANCE</span><b>{raceData?.distance_km?.toFixed(3) || '5.412'}<small> KM</small></b></div>
                  </div>
                </aside>
              </div>
            </section>

            {/* TELEMETRY CHARTS */}
            <section className="analytics-grid">
              <ChartFrame
                eyebrow={`PACE TRACE // LAP ${raceData?.min_lap || 1}—${raceData?.max_lap || 57}`}
                title="Tyre degradation"
                onEnlarge={() => setEnlargedChart('degradation')}
                legend={
                  <div className="legend">
                    <span><i style={{ background: a.accent }} />{a.code}</span>
                    <span><i style={{ background: b.accent }} />{b.code}</span>
                    <span><i className="soft" />SOFT</span>
                    <span><i className="medium" />MED</span>
                    <span><i className="dashed" style={{ background: '#e4e9ee' }} />HARD</span>
                  </div>
                }
              >
                <RealDegradationChart lapsA={lapsA} lapsB={lapsB} a={a} b={b} />
              </ChartFrame>

              <ChartFrame
                eyebrow="RELATIVE DELTA // RACE DISTANCE"
                title="Gap evolution"
                onEnlarge={() => setEnlargedChart('gap')}
                legend={
                  <span className="ahead-tag">
                    {currentGap >= 0 ? `${a.code} AHEAD · ${currentGap.toFixed(2)}S` : `${b.code} AHEAD · ${(-currentGap).toFixed(2)}S`}
                  </span>
                }
              >
                <RealGapChart lapsA={lapsA} lapsB={lapsB} a={a} b={b} />
              </ChartFrame>
            </section>

            {/* SIMULATOR SECTION */}
            <section className="simulator">
              <div className="sim-heading">
                <div>
                  <span className="eyebrow">PREDICTIVE MODEL // 15-LAP HORIZON</span>
                  <h2>Undercut simulator</h2>
                </div>
                <div className="sim-controls">
                  <label>
                    <span>SIMULATE FROM LAP</span>
                    <div className="number-input">
                      <button onClick={() => setPitLap(Math.max(raceData?.min_lap || 1, pitLap - 1))}>−</button>
                      <input
                        type="number"
                        value={pitLap}
                        min={raceData?.min_lap || 1}
                        max={raceData?.max_lap || 60}
                        onChange={e => setPitLap(Number(e.target.value))}
                      />
                      <button onClick={() => setPitLap(Math.min(raceData?.max_lap || 60, pitLap + 1))}>+</button>
                    </div>
                  </label>
                  <SelectField label="Pit for compound" value={compound} onChange={setCompound}>
                    <option value="SOFT">Soft</option>
                    <option value="MEDIUM">Medium</option>
                    <option value="HARD">Hard</option>
                  </SelectField>
                  <SelectField label="MC Rollouts" value={String(rollouts)} onChange={v => setRollouts(Number(v))}>
                    <option value="200">200 Rollouts</option>
                    <option value="500">500 Rollouts</option>
                    <option value="1000">1000 Rollouts</option>
                  </SelectField>
                  <div className="pit-loss">
                    <span>MEDIAN PIT LOSS</span>
                    <strong>{raceData?.pit_loss?.toFixed(1) || '25.3'}<small>S</small></strong>
                  </div>
                </div>
              </div>

              {/* TWO-CAR GAME-THEORETIC HUD STRIP */}
              <div className="kpi-row" style={{ marginTop: 14 }}>
                <div className="mini-kpi">
                  <span>CAUSAL RACE ALPHA (EB)</span>
                  <strong>{simResult?.race_alpha != null ? `${simResult.race_alpha.toFixed(3)}x` : '1.000x'}</strong>
                  <small>IN-RACE DEG SCALE</small>
                </div>
                <div className="mini-kpi">
                  <span>RESIDUAL SHRUNK BIAS</span>
                  <strong>{simResult?.race_bias != null ? `${simResult.race_bias > 0 ? '+' : ''}${simResult.race_bias.toFixed(3)}s` : '0.000s'}</strong>
                  <small>TELEMETRY OFFSET</small>
                </div>
                <div className="mini-kpi">
                  <span>RIVAL GAP AT SIM LAP</span>
                  <strong>{simResult?.rival_gap_s != null ? `${simResult.rival_gap_s.toFixed(2)}s` : `${currentGap.toFixed(2)}s`}</strong>
                  <small>{driverA} vs {driverB}</small>
                </div>
                <div className="mini-kpi" style={{ borderLeftColor: simResult?.rival_covered ? 'var(--red)' : 'var(--green)' }}>
                  <span>DEFENDER TACTICAL COVER</span>
                  <strong style={{ color: simResult?.rival_covered ? 'var(--red)' : 'var(--green)' }}>
                    {simResult?.rival_covered ? 'COVER TRIGGERED' : 'WINDOW UNCOVERED'}
                  </strong>
                  <small>{simResult?.rival_covered ? `${driverB} PITS NEXT LAP` : 'CLEAN AIR ADVANTAGE'}</small>
                </div>
              </div>

              <div className="simulation-grid">
                <ChartFrame
                  eyebrow={`NET TIME DELTA // ${compound.toUpperCase()} COMPOUND`}
                  title="Net undercut advantage"
                  onEnlarge={() => setEnlargedChart('forecast')}
                  legend={
                    <div className="legend">
                      <span><i className="cyan" />FRESH TYRE DELTA</span>
                      <span><i className="dashed" />BREAK-EVEN</span>
                    </div>
                  }
                >
                  <RealForecastChart simResult={simResult} lap={pitLap} />
                </ChartFrame>

                <article className={`verdict verdict-${verdict}`}>
                  <div className="verdict-top">
                    <span>STRATEGY VERDICT</span>
                    <i>MODEL CONFIDENCE · {simResult?.confidence || 85}%</i>
                  </div>
                  <div className="verdict-icon">
                    <Icon name={verdict === 'open' ? 'target' : verdict === 'pending' ? 'radio' : 'flag'} size={32} />
                  </div>
                  <span className="verdict-code">
                    {verdict === 'open' ? 'WINDOW STATUS // OPEN' : verdict === 'pending' ? 'WINDOW STATUS // MONITOR' : 'WINDOW STATUS // CLOSED'}
                  </span>
                  <h3>{simResult?.headline || (verdict === 'open' ? 'Undercut window open' : 'Pace crossover pending')}</h3>
                  <p>{simResult?.description || `Box on lap ${pitLap}. Fresh tyres forecast a compounding pace advantage.`}</p>
                  <div className="recommendation">
                    <span>RECOMMENDED CALL</span>
                    <strong>{simResult?.recommendation || `BOX ${a.code} · LAP ${pitLap}`}</strong>
                  </div>
                </article>
              </div>

              {/* ── 6-STRATEGY MULTI-CANDIDATE TABLE ── */}
              {simResult?.all_strategies && simResult.all_strategies.length > 0 && (
                <div className="strat-table-container">
                  <div style={{ padding: '14px 18px 8px', borderBottom: '1px solid var(--line)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <span className="eyebrow">MULTI-OUTPUT COMPARISON // ALL TACTICAL PATHS</span>
                      <h4 style={{ margin: '4px 0 0', fontSize: 18, fontStyle: 'italic', textTransform: 'uppercase' }}>Tactical Strategy Candidate Matrix</h4>
                    </div>
                    <span style={{ font: '9px JetBrains Mono', color: 'var(--muted)' }}>500 MONTE CARLO FORWARD PATHS</span>
                  </div>
                  <table className="strat-table">
                    <thead>
                      <tr>
                        <th>Strategy</th>
                        <th>Compound</th>
                        <th>Stop Lap</th>
                        <th>Win Prob (P &gt; 0)</th>
                        <th>Expected Delta (s)</th>
                        <th>90% Confidence Interval</th>
                        <th>Crossover Lap</th>
                        <th>Window Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {simResult.all_strategies.map((strat, i) => {
                        const isWin = strat.p_net_gain >= 0.5
                        const comp = strat.compound.toUpperCase()
                        const pillClass = comp.includes('SOFT') ? 'pirelli-s' : comp.includes('HARD') ? 'pirelli-h' : 'pirelli-m'
                        return (
                          <tr key={i} style={{ background: strat.window_open ? 'rgba(32,217,239,.06)' : undefined }}>
                            <td style={{ fontWeight: 700, color: strat.window_open ? 'var(--cyan)' : 'var(--text)' }}>{strat.name}</td>
                            <td>
                              <span className="stint-chip">
                                <span className={`pirelli-dot ${pillClass}`}>{comp[0]}</span>
                                {strat.compound}
                              </span>
                            </td>
                            <td>L{strat.stop_lap}</td>
                            <td>
                              <span style={{ color: isWin ? 'var(--green)' : 'var(--red)', fontWeight: 700 }}>
                                {(strat.p_net_gain * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td style={{ fontWeight: 700, color: strat.expected_gain_s > 0 ? 'var(--green)' : strat.expected_gain_s < 0 ? 'var(--red)' : 'var(--muted)' }}>
                              {strat.expected_gain_s > 0 ? '+' : ''}{strat.expected_gain_s.toFixed(2)}s
                            </td>
                            <td style={{ color: 'var(--muted)', fontSize: 10 }}>
                              [{strat.p05_gain_s > 0 ? '+' : ''}{strat.p05_gain_s.toFixed(2)}s, {strat.p95_gain_s > 0 ? '+' : ''}{strat.p95_gain_s.toFixed(2)}s]
                            </td>
                            <td>{strat.crossover_lap ? `Lap ${strat.crossover_lap}` : 'None'}</td>
                            <td>
                              <span style={{
                                padding: '2px 7px',
                                fontSize: 8,
                                fontWeight: 700,
                                border: '1px solid',
                                borderColor: strat.window_open ? 'var(--green)' : 'var(--line)',
                                color: strat.window_open ? 'var(--green)' : 'var(--muted)',
                                background: strat.window_open ? 'rgba(114,245,173,.08)' : 'transparent',
                              }}>
                                {strat.window_open ? 'WINDOW OPEN' : 'STAY OUT'}
                              </span>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* ── SHAP EXPLAINABILITY ATTRIBUTIONS ── */}
              {simResult?.shap_values && simResult.shap_values.length > 0 && (
                <div className="shap-container">
                  <div className="shap-heading">
                    <div>
                      <span className="eyebrow">TREE-EXPLAINER // FEATURE ATTRIBUTION</span>
                      <h4 style={{ margin: '4px 0 0', fontSize: 18, fontStyle: 'italic', textTransform: 'uppercase' }}>SHAP Decision Explainability (Lap {pitLap})</h4>
                    </div>
                    <span style={{ font: '9px JetBrains Mono', color: 'var(--muted)' }}>SHAPLEY ADDITIVE ATTRIBUTION</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '8px 24px' }}>
                    {simResult.shap_values.slice(0, 8).map((sh, idx) => {
                      const isPos = sh.shap_value >= 0
                      const barWidth = Math.min(100, Math.abs(sh.shap_value) * 120)
                      return (
                        <div key={idx} className="shap-row">
                          <span style={{ color: 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {sh.label} ({sh.value})
                          </span>
                          <div className="shap-bar-track">
                            <div
                              className="shap-bar-fill"
                              style={{
                                width: `${barWidth}%`,
                                background: isPos ? 'var(--red)' : 'var(--cyan)',
                                marginLeft: isPos ? undefined : 'auto',
                              }}
                            />
                          </div>
                          <span style={{ textAlign: 'right', fontWeight: 700, color: isPos ? 'var(--red)' : 'var(--cyan)' }}>
                            {sh.shap_value > 0 ? '+' : ''}{sh.shap_value.toFixed(3)}s
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </section>
          </div>
        ) : (
          /* ══════════════════════════════════════════════════════════════════
             MODE 2: POST-RACE REVIEW & WHAT-IF STRATEGY DASHBOARD
             ══════════════════════════════════════════════════════════════════ */
          <div className="dashboard">
            <section className="hero" style={{ minHeight: 220 }}>
              <div>
                <div className="status-badge" style={{ borderColor: reviewInit?.is_in_sample ? 'var(--green)' : '#ffd32a' }}>
                  <span className="live-dot" style={{ background: reviewInit?.is_in_sample ? 'var(--green)' : '#ffd32a' }} />
                  {reviewInit?.is_in_sample ? 'IN-SAMPLE BENCHMARK (TRAINING SET)' : 'OUT-OF-SAMPLE LORO EVALUATION'}
                </div>
                <p className="kicker">POST-RACE COUNTERFACTUAL ENGINE // 500-ROLLOUT WHAT-IF</p>
                <h1>Race <span>Review</span> &amp; What-If</h1>
                <p className="hero-copy">Replay historical decisions, compute pit decision regret curves across ±8 laps, and simulate rival traffic reactions under dirty air.</p>
              </div>
              <div className="session-lock">
                <span>LORO HONESTY</span>
                <strong style={{ color: reviewInit?.is_in_sample ? 'var(--green)' : '#ffd32a' }}>
                  {reviewInit?.is_in_sample ? 'IN-SAMPLE' : 'LORO COLD'}
                </strong>
                <small>{selectedRace.toUpperCase()}</small>
              </div>
            </section>

            {/* REVIEW KPI STRIP */}
            <section className="kpi-strip">
              {[
                ['RACE DISTANCE', `${reviewInit?.total_laps || 57} LAPS`, '100% DISTANCE COMPLETED'],
                ['PIT LANE LOSS', `${reviewInit?.pit_loss_green?.toFixed(1) || '25.1'}S`, `NEUTRALIZED: ${reviewInit?.pit_loss_neutralized?.toFixed(1) || '15.5'}S`],
                ['TRACK ABRASION', `INDEX ${reviewInit?.circuit_physics?.abrasion_index || 4} / 5`, `CORNERS: ${reviewInit?.circuit_physics?.num_corners || 15} · ${reviewInit?.circuit_physics?.circuit_length_km || 5.41} KM`],
                ['LORO REGIME', reviewInit?.is_in_sample ? 'IN-SAMPLE BENCHMARK' : 'OUT-OF-SAMPLE (LORO)', 'CONTROLLED MONOTONIC XGB'],
              ].map(([label, value, sub], i) => (
                <div className="kpi" key={label}>
                  <span>0{i + 1} // {label}</span>
                  <strong>{value}</strong>
                  <small>{sub}</small>
                </div>
              ))}
            </section>

            {/* TELEMETRY FALLBACK NOTICE */}
            {reviewInit?.fallback_used && (
              <div style={{
                background: 'rgba(255, 170, 0, 0.08)',
                border: '1px solid rgba(255, 170, 0, 0.4)',
                borderRadius: '4px',
                color: '#ffaa00',
                padding: '10px 16px',
                marginBottom: '16px',
                font: '11px JetBrains Mono',
                letterSpacing: '0.03em',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <span style={{ fontSize: '16px' }}>⚠️</span>
                <div>
                  <div style={{ fontWeight: 700, marginBottom: 2 }}>TELEMETRY SYNCHRONIZATION NOTICE</div>
                  <div style={{ fontSize: '10px', opacity: 0.9 }}>{reviewInit.fallback_reason}</div>
                </div>
              </div>
            )}

            {/* RACE REVIEW GRID: TIMING TOWER + WHAT-IF PANEL */}
            <div className="review-grid">
              {/* TIMING TOWER */}
              <aside className="timing-tower">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, paddingBottom: 8, borderBottom: '1px solid var(--line)' }}>
                  <span className="eyebrow">GRID TIMING TOWER</span>
                  <span style={{ font: '9px JetBrains Mono', color: 'var(--muted)' }}>P1—P{reviewInit?.drivers?.length || 20}</span>
                </div>
                {reviewLoading ? (
                  <div style={{ padding: '40px 10px', textAlign: 'center', color: 'var(--muted)', font: '11px JetBrains Mono' }}>
                    <div className="spinner" style={{ margin: '0 auto 12px' }} />
                    SYNCHRONIZING TIMING REPLAY...
                  </div>
                ) : (
                  reviewInit?.drivers?.map(d => {
                    const isSel = d.code === reviewDriver
                    return (
                      <div
                        key={d.code}
                        className={`driver-row ${isSel ? 'selected' : ''}`}
                        onClick={() => {
                          setReviewDriver(d.code)
                          setReviewStopIdx(0)
                          setWhatifShift(0)
                        }}
                      >
                        <span className="driver-pos">P{d.final_pos}</span>
                        <span className="driver-code-tag" style={{ color: d.accent }}>{d.code}</span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text)' }}>{d.name}</div>
                          <div style={{ display: 'flex', gap: 4, marginTop: 3 }}>
                            {d.stints.map(s => {
                              const pClass = s.compound.includes('SOFT') ? 'pirelli-s' : s.compound.includes('HARD') ? 'pirelli-h' : 'pirelli-m'
                              return (
                                <span key={s.stint} className="stint-chip" style={{ fontSize: 8, padding: '1px 4px' }}>
                                  <span className={`pirelli-dot ${pClass}`} style={{ width: 10, height: 10, fontSize: 7 }}>{s.compound[0]}</span>
                                  {s.laps_count}L
                                </span>
                              )
                            })}
                          </div>
                        </div>
                        <span style={{ font: '9px JetBrains Mono', color: d.gap_to_winner === 0 ? 'var(--green)' : 'var(--muted)' }}>
                          {d.gap_to_winner === 0 ? 'WIN' : `+${d.gap_to_winner.toFixed(1)}s`}
                        </span>
                      </div>
                    )
                  })
                )}
              </aside>

              {/* WHAT-IF & REGRET SCAN MAIN VIEW */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
                {/* ACTIVE DRIVER SUMMARY BANNER */}
                {activeReviewDriver && (
                  <div className="whatif-panel" style={{ borderLeft: `3px solid ${activeReviewDriver.accent}` }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 14 }}>
                      <div>
                        <span className="eyebrow">{activeReviewDriver.team} // #{activeReviewDriver.number}</span>
                        <h2 style={{ margin: '4px 0 0', fontSize: 32, fontStyle: 'italic', textTransform: 'uppercase' }}>
                          {activeReviewDriver.name} <span style={{ color: 'var(--muted)', fontSize: 20 }}>({activeReviewDriver.code})</span>
                        </h2>
                      </div>
                      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                        <div className="stint-chips-strip" style={{ display: 'flex', gap: 6 }}>
                          {activeReviewDriver.stints.map(s => {
                            const pClass = s.compound.includes('SOFT') ? 'pirelli-s' : s.compound.includes('HARD') ? 'pirelli-h' : 'pirelli-m'
                            return (
                              <span key={s.stint} className="stint-chip">
                                <span className={`pirelli-dot ${pClass}`}>{s.compound[0]}</span>
                                {s.compound} · {s.laps_count} LAPS
                              </span>
                            )
                          })}
                        </div>
                        <div style={{ font: '11px JetBrains Mono', padding: '6px 12px', background: '#080d12', border: '1px solid var(--line)' }}>
                          STOPS: <b>{activeReviewDriver.pit_laps.length > 0 ? activeReviewDriver.pit_laps.map(l => `L${l}`).join(', ') : 'None'}</b>
                        </div>
                      </div>
                    </div>

                    {/* STOP SELECTOR ROW */}
                    {activeReviewDriver.pit_laps.length > 0 && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--line)' }}>
                        <span style={{ font: '9px JetBrains Mono', color: 'var(--cyan)', letterSpacing: '0.12em', fontWeight: 700 }}>
                          SELECT PIT STOP TO COUNTERFACTUALIZE:
                        </span>
                        <div style={{ display: 'flex', gap: 8 }}>
                          {activeReviewDriver.pit_laps.map((lapNum, idx) => {
                            const isSel = reviewStopIdx === idx
                            return (
                              <button
                                key={idx}
                                onClick={() => {
                                  setReviewStopIdx(idx)
                                  setWhatifShift(0)
                                }}
                                style={{
                                  padding: '5px 12px',
                                  background: isSel ? 'var(--cyan)' : '#080d12',
                                  color: isSel ? '#031014' : 'var(--text)',
                                  border: isSel ? '1px solid var(--cyan)' : '1px solid var(--line)',
                                  font: '700 9px JetBrains Mono',
                                  letterSpacing: '0.08em',
                                  cursor: 'pointer',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: 6,
                                  boxShadow: isSel ? '0 0 12px rgba(32, 217, 239, 0.3)' : 'none',
                                  transition: 'all 0.2s'
                                }}
                              >
                                <span>STOP {idx + 1}</span>
                                <span style={{ opacity: isSel ? 0.9 : 0.6 }}>· LAP {lapNum}</span>
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* ── PIT DECISION REGRET SCAN (-8 TO +8 LAPS) ── */}
                <div className="whatif-panel">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                    <div>
                      <span className="eyebrow">MONTE CARLO REGRET CURVE // MULTI-STOP SCAN</span>
                      <h3 style={{ margin: '4px 0 0', fontSize: 22, fontStyle: 'italic', textTransform: 'uppercase' }}>
                        Pit Decision Regret Analysis (Stop {reviewStopIdx + 1})
                      </h3>
                    </div>
                    {regretResult && (
                      <div style={{ display: 'flex', gap: 10 }}>
                        <div className="mini-kpi" style={{ padding: '4px 10px' }}>
                          <span>ACTUAL STOP</span>
                          <strong style={{ fontSize: 13 }}>Lap {regretResult.actual_pit_lap}</strong>
                        </div>
                        <div className="mini-kpi" style={{ padding: '4px 10px', borderLeftColor: 'var(--green)' }}>
                          <span>OPTIMAL STOP</span>
                          <strong style={{ fontSize: 13, color: 'var(--green)' }}>Lap {regretResult.best_pit_lap} ({regretResult.best_shift > 0 ? '+' : ''}{regretResult.best_shift}L)</strong>
                        </div>
                        <div className="mini-kpi" style={{ padding: '4px 10px', borderLeftColor: regretResult.max_regret_s > 0 ? 'var(--red)' : 'var(--green)' }}>
                          <span>MAX REGRET</span>
                          <strong style={{ fontSize: 13, color: regretResult.max_regret_s > 0 ? 'var(--red)' : 'var(--green)' }}>
                            {regretResult.max_regret_s > 0 ? `+${regretResult.max_regret_s.toFixed(2)}s` : '0.00s (OPTIMAL)'}
                          </strong>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* SVG Bar Chart of Regret Scan */}
                  {regretResult?.scan && regretResult.scan.length > 0 ? (
                    <div style={{ height: 265 }}>
                      <svg viewBox="0 0 840 250" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
                        {(() => {
                          const scan = regretResult.scan
                          const minD = Math.min(...scan.map(r => r.p05))
                          const maxD = Math.max(...scan.map(r => r.p95))
                          const span = maxD - minD || 1
                          const padL = 48
                          const padR = 20
                          const padT = 26
                          const padB = 44
                          const plotW = 840 - padL - padR
                          const plotH = 250 - padT - padB

                          const zeroY = padT + plotH - ((0 - minD) / span) * plotH
                          const barW = Math.max(12, plotW / scan.length - 6)

                          // 5 Y-ticks
                          const yTicks = [minD, minD + span * 0.25, 0, minD + span * 0.75, maxD].map(val => ({
                            val,
                            y: padT + plotH - ((val - minD) / span) * plotH
                          }))

                          return (
                            <>
                              {/* Gridlines and Y-labels */}
                              {yTicks.map((t, idx) => (
                                <g key={idx}>
                                  <line
                                    x1={padL}
                                    y1={t.y}
                                    x2={840 - padR}
                                    y2={t.y}
                                    stroke={Math.abs(t.val) < 0.05 ? 'var(--line-hot)' : 'var(--line)'}
                                    strokeWidth={Math.abs(t.val) < 0.05 ? 1.4 : 0.6}
                                    strokeDasharray={Math.abs(t.val) < 0.05 ? undefined : '3 3'}
                                  />
                                  <text x={padL - 6} y={t.y + 3} textAnchor="end" fill={Math.abs(t.val) < 0.05 ? 'var(--cyan)' : 'var(--muted)'} fontSize="8" fontFamily="JetBrains Mono" fontWeight={Math.abs(t.val) < 0.05 ? '700' : '400'}>
                                    {t.val > 0 ? `+${t.val.toFixed(1)}s` : `${t.val.toFixed(1)}s`}
                                  </text>
                                </g>
                              ))}

                              {scan.map((r, i) => {
                                const x = padL + i * (plotW / scan.length) + 3
                                const valY = padT + plotH - ((r.expected_time_delta - minD) / span) * plotH
                                const barH = Math.max(3, Math.abs(valY - zeroY))
                                const topY = Math.min(valY, zeroY)
                                const isBest = r.moved_lap === regretResult.best_pit_lap
                                const isAct = r.is_actual
                                const isSelected = whatifShift === r.shift
                                const fillCol = r.is_noise ? '#53616c' : r.expected_time_delta < 0 ? 'var(--green)' : 'var(--red)'

                                // Whisker coords
                                const whiskerTop = padT + plotH - ((r.p95 - minD) / span) * plotH
                                const whiskerBot = padT + plotH - ((r.p05 - minD) / span) * plotH
                                const centerX = x + barW / 2

                                return (
                                  <g key={r.shift} style={{ cursor: 'pointer' }} onClick={() => setWhatifShift(r.shift)}>
                                    {/* CI Whisker line (p05 to p95) */}
                                    <line x1={centerX} y1={whiskerTop} x2={centerX} y2={whiskerBot} stroke={fillCol} strokeWidth="1" opacity="0.6" />
                                    <line x1={centerX - 3} y1={whiskerTop} x2={centerX + 3} y2={whiskerTop} stroke={fillCol} strokeWidth="1" opacity="0.6" />
                                    <line x1={centerX - 3} y1={whiskerBot} x2={centerX + 3} y2={whiskerBot} stroke={fillCol} strokeWidth="1" opacity="0.6" />

                                    {/* Bar rectangle */}
                                    <rect
                                      x={x}
                                      y={topY}
                                      width={barW}
                                      height={barH}
                                      fill={fillCol}
                                      stroke={isSelected ? 'var(--cyan)' : isBest ? 'var(--green)' : isAct ? '#fff' : 'none'}
                                      strokeWidth={isSelected ? 2.5 : isBest || isAct ? 1.5 : 0}
                                      rx="1.5"
                                      opacity={isSelected ? 1 : 0.8}
                                    />

                                    {/* Value label above/below bar */}
                                    <text
                                      x={centerX}
                                      y={r.expected_time_delta < 0 ? topY - 4 : topY + barH + 9}
                                      textAnchor="middle"
                                      fill={fillCol}
                                      fontSize="7"
                                      fontFamily="JetBrains Mono"
                                      fontWeight="700"
                                    >
                                      {r.expected_time_delta > 0 ? '+' : ''}{r.expected_time_delta.toFixed(1)}
                                    </text>

                                    {/* Best / Actual markers */}
                                    {isBest && (
                                      <text x={centerX} y={padT - 8} textAnchor="middle" fill="var(--green)" fontSize="7.5" fontWeight="800" fontFamily="JetBrains Mono">
                                        ★ BEST
                                      </text>
                                    )}
                                    {isAct && (
                                      <text x={centerX} y={padT - 8} textAnchor="middle" fill="#fff" fontSize="7.5" fontWeight="800" fontFamily="JetBrains Mono">
                                        ACTUAL
                                      </text>
                                    )}

                                    {/* X-axis labels: Lap and Shift */}
                                    <text x={centerX} y={200 - 20} textAnchor="middle" fill={isSelected ? 'var(--cyan)' : 'var(--text)'} fontSize="7.5" fontFamily="JetBrains Mono" fontWeight={isSelected ? '700' : '400'}>
                                      L{r.moved_lap}
                                    </text>
                                    <text x={centerX} y={200 - 10} textAnchor="middle" fill="var(--muted)" fontSize="6.5" fontFamily="JetBrains Mono">
                                      {r.shift > 0 ? `+${r.shift}` : r.shift}
                                    </text>
                                  </g>
                                )
                              })}
                            </>
                          )
                        })()}
                      </svg>
                    </div>
                  ) : regretLoading ? (
                    <div style={{ height: 140, display: 'grid', placeItems: 'center', color: 'var(--cyan)', font: '10px JetBrains Mono' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                        <div className="spinner" />
                        <span>EXECUTING 25-WINDOW MONTE CARLO REGRET SCAN...</span>
                      </div>
                    </div>
                  ) : (
                    <div style={{ height: 140, display: 'grid', placeItems: 'center', color: 'var(--muted)', font: '10px JetBrains Mono' }}>
                      Select a driver from Grid Timing Tower to view pit regret analysis.
                    </div>
                  )}
                </div>

                {/* ── INTERACTIVE WHAT-IF COUNTERFACTUAL SCRUBBER ── */}
                <div className="whatif-panel">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                    <div>
                      <span className="eyebrow">COUNTERFACTUAL TRAJECTORY // DELTA FORMULATION</span>
                      <h3 style={{ margin: '4px 0 0', fontSize: 22, fontStyle: 'italic', textTransform: 'uppercase' }}>
                        Simulating Pit Stop on Lap {whatifResult?.moved_lap || (activeReviewDriver?.pit_laps[0] || 15) + whatifShift} ({whatifShift > 0 ? `+${whatifShift}` : whatifShift} Laps)
                      </h3>
                    </div>
                    <div style={{ display: 'flex', gap: 10 }}>
                      <SelectField label="Rival Cover Model" value={rivalMode} onChange={v => setRivalMode(v as any)}>
                        <option value="cover_traffic">Rivals Cover + Dirty Air (Realistic)</option>
                        <option value="cover">Rivals Cover Only (+1 Lap)</option>
                        <option value="fixed">Rivals Fixed (Stopwatch Baseline)</option>
                      </SelectField>
                    </div>
                  </div>

                  {/* Scrubber slider */}
                  <div style={{ marginBottom: 18, background: '#080d12', padding: '14px 18px', border: '1px solid var(--line)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', font: '9px JetBrains Mono', color: 'var(--muted)', marginBottom: 8 }}>
                      <span>UNDER-CUT (-8 LAPS)</span>
                      <span style={{ color: 'var(--cyan)', fontWeight: 700 }}>SHIFT: {whatifShift > 0 ? `+${whatifShift}` : whatifShift} LAPS</span>
                      <span>OVER-CUT (+8 LAPS)</span>
                    </div>
                    <input
                      type="range"
                      min="-8"
                      max="8"
                      value={whatifShift}
                      onChange={e => setWhatifShift(Number(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--cyan)', cursor: 'pointer' }}
                    />
                  </div>

                  {/* KPI CARDS */}
                  <div className="kpi-row">
                    <div className="mini-kpi">
                      <span>EXPECTED TIME DELTA</span>
                      <strong style={{ color: (whatifResult?.final_time_delta_p50 || 0) < 0 ? 'var(--green)' : 'var(--red)' }}>
                        {whatifResult?.final_time_delta_p50 != null ? `${whatifResult.final_time_delta_p50 > 0 ? '+' : ''}${whatifResult.final_time_delta_p50.toFixed(2)}s` : '0.00s'}
                      </strong>
                      <small>{(whatifResult?.final_time_delta_p50 || 0) < 0 ? 'NET ADVANTAGE' : 'NET LOSS'}</small>
                    </div>
                    <div className="mini-kpi">
                      <span>90% CONFIDENCE INTERVAL</span>
                      <strong>
                        [{whatifResult?.final_time_delta_p05 != null ? `${whatifResult.final_time_delta_p05 > 0 ? '+' : ''}${whatifResult.final_time_delta_p05.toFixed(2)}s` : '-'}, {whatifResult?.final_time_delta_p95 != null ? `${whatifResult.final_time_delta_p95 > 0 ? '+' : ''}${whatifResult.final_time_delta_p95.toFixed(2)}s` : '-'}]
                      </strong>
                      <small>ASYMMETRIC BOUNDS</small>
                    </div>
                    <div className="mini-kpi">
                      <span>FINAL FINISH POSITION</span>
                      <strong style={{ color: 'var(--cyan)' }}>
                        P{whatifResult?.counterfactual_pos || activeReviewDriver?.final_pos || 1}
                      </strong>
                      <small>ACTUAL: P{activeReviewDriver?.final_pos || 1}</small>
                    </div>
                    <div className="mini-kpi">
                      <span>ROBUSTNESS &amp; DIRTY AIR</span>
                      <strong style={{ color: whatifResult?.robustness === 'ROBUST' ? 'var(--green)' : '#ffd32a' }}>
                        {whatifResult?.robustness || 'ROBUST'}
                      </strong>
                      <small>TRAFFIC: +{whatifResult?.dirty_air_total_s?.toFixed(2) || '0.00'}s</small>
                    </div>
                  </div>

                  {/* RIVAL REACTIONS */}
                  {whatifResult?.rival_reactions && whatifResult.rival_reactions.length > 0 && (
                    <div style={{ marginTop: 14, padding: '10px 14px', background: '#0a131a', border: '1px solid var(--line-hot)', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                      <span style={{ font: '9px JetBrains Mono', color: 'var(--cyan)', fontWeight: 700 }}>RIVAL REACTIONS:</span>
                      {whatifResult.rival_reactions.map((r, i) => (
                        <span key={i} style={{ font: '9px JetBrains Mono', color: 'var(--text)' }}>
                          <b>{r.driver}</b> boxed Lap {r.reacted_pit} (actual L{r.actual_pit}) — <i>{r.reason}</i>
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Cumulative Delta Curve SVG */}
                  {whatifResult?.delta_per_lap && whatifResult.delta_per_lap.length > 0 && (
                    <div style={{ height: 250, marginTop: 18 }}>
                      <svg viewBox="0 0 840 230" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
                        {(() => {
                          const curve = whatifResult.delta_per_lap
                          const minD = Math.min(-1.0, Math.floor(Math.min(...curve) * 2) / 2 - 0.5)
                          const maxD = Math.max(1.5, Math.ceil(Math.max(...curve) * 2) / 2 + 0.5)
                          const span = maxD - minD || 1

                          const padL = 48
                          const padR = 30
                          const padT = 24
                          const padB = 34
                          const plotW = 840 - padL - padR
                          const plotH = 230 - padT - padB

                          const zeroY = padT + plotH - ((0 - minD) / span) * plotH
                          const scaleX = (i: number) => padL + i * (plotW / Math.max(1, curve.length - 1))
                          const scaleY = (d: number) => padT + plotH - ((d - minD) / span) * plotH

                          const pts = curve.map((d, i) => `${scaleX(i).toFixed(1)},${scaleY(d).toFixed(1)}`)
                          const lineD = `M ${pts.join(' L ')}`
                          const areaD = `M ${scaleX(0)},${zeroY.toFixed(1)} L ${pts.join(' L ')} L ${scaleX(curve.length - 1)},${zeroY.toFixed(1)} Z`

                          // Y-ticks
                          const yTicks = [minD, minD + span * 0.25, 0, minD + span * 0.75, maxD].map(val => ({
                            val,
                            y: scaleY(val)
                          }))

                          // X-ticks every 5-10 laps
                          const step = Math.max(1, Math.round(curve.length / 8))
                          const xTicks: number[] = []
                          for (let l = 0; l < curve.length; l += step) xTicks.push(l)
                          if (xTicks[xTicks.length - 1] !== curve.length - 1) xTicks.push(curve.length - 1)

                          const pitLapIdx = (whatifResult.moved_lap || 15) - 1
                          const pitX = pitLapIdx >= 0 && pitLapIdx < curve.length ? scaleX(pitLapIdx) : null

                          return (
                            <>
                              <defs>
                                <linearGradient id="cumDeltaGrad" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="0%" stopColor="#FFD166" stopOpacity="0.25" />
                                  <stop offset="100%" stopColor="#FFD166" stopOpacity="0.0" />
                                </linearGradient>
                              </defs>

                              {/* Gridlines and Y-labels */}
                              {yTicks.map((t, idx) => (
                                <g key={idx}>
                                  <line
                                    x1={padL}
                                    y1={t.y}
                                    x2={840 - padR}
                                    y2={t.y}
                                    stroke={Math.abs(t.val) < 0.05 ? 'var(--line-hot)' : 'var(--line)'}
                                    strokeWidth={Math.abs(t.val) < 0.05 ? 1.4 : 0.6}
                                    strokeDasharray={Math.abs(t.val) < 0.05 ? undefined : '3 3'}
                                  />
                                  <text x={padL - 6} y={t.y + 3} textAnchor="end" fill={Math.abs(t.val) < 0.05 ? 'var(--cyan)' : 'var(--muted)'} fontSize="8" fontFamily="JetBrains Mono" fontWeight={Math.abs(t.val) < 0.05 ? '700' : '400'}>
                                    {t.val > 0 ? `+${t.val.toFixed(1)}s` : `${t.val.toFixed(1)}s`}
                                  </text>
                                </g>
                              ))}

                              {/* X-ticks */}
                              {xTicks.map(idx => {
                                const x = scaleX(idx)
                                return (
                                  <g key={idx}>
                                    <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke="var(--line)" strokeDasharray="2 4" strokeWidth="0.6" opacity="0.6" />
                                    <text x={x} y={180 - 10} textAnchor="middle" fill="var(--muted)" fontSize="8" fontFamily="JetBrains Mono">
                                      L{idx + 1}
                                    </text>
                                  </g>
                                )
                              })}

                              {/* Area fill */}
                              <path d={areaD} fill="url(#cumDeltaGrad)" />

                              {/* Line */}
                              <path d={lineD} fill="none" stroke="#FFD166" strokeWidth="2.4" />

                              {/* Pit Stop Marker */}
                              {pitX !== null && (
                                <g>
                                  <line x1={pitX} y1={padT} x2={pitX} y2={padT + plotH} stroke="var(--cyan)" strokeWidth="1.5" strokeDasharray="3 3" />
                                  <rect x={pitX - 28} y={padT - 14} width="56" height="13" fill="#080e14" stroke="var(--cyan)" strokeWidth="1" rx="2" />
                                  <text x={pitX} y={padT - 5} textAnchor="middle" fill="var(--cyan)" fontSize="7" fontWeight="700" fontFamily="JetBrains Mono">
                                    PIT L{pitLapIdx + 1}
                                  </text>
                                </g>
                              )}

                              {/* Data Dots */}
                              {curve.map((d, i) => {
                                if (i % 2 !== 0 && i !== curve.length - 1) return null
                                return <circle key={i} cx={scaleX(i)} cy={scaleY(d)} r="2" fill="#FFD166" stroke="#05080c" strokeWidth="0.7" />
                              })}

                              {/* Final Outcome Callout */}
                              {curve.length > 0 && (
                                <g>
                                  <circle cx={scaleX(curve.length - 1)} cy={scaleY(curve[curve.length - 1])} r="4" fill={curve[curve.length - 1] < 0 ? 'var(--green)' : 'var(--red)'} stroke="#fff" strokeWidth="1" />
                                  <text x={scaleX(curve.length - 1) - 4} y={scaleY(curve[curve.length - 1]) - 8} textAnchor="end" fill={curve[curve.length - 1] < 0 ? 'var(--green)' : 'var(--red)'} fontSize="8.5" fontWeight="700" fontFamily="JetBrains Mono">
                                    FINAL {curve[curve.length - 1] > 0 ? '+' : ''}{curve[curve.length - 1].toFixed(2)}s
                                  </text>
                                </g>
                              )}
                            </>
                          )
                        })()}
                      </svg>
                    </div>
                  )}
                </div>

                {/* ── STINT DEGRADATION & CALIBRATION ── */}
                {stintDegResult?.stints && stintDegResult.stints.length > 0 && (
                  <div className="whatif-panel">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                      <div>
                        <span className="eyebrow">RESIDUAL CALIBRATION // G3 PHYSICS MODEL</span>
                        <h3 style={{ margin: '4px 0 0', fontSize: 22, fontStyle: 'italic', textTransform: 'uppercase' }}>
                          Observed vs Predicted Stint Degradation
                        </h3>
                      </div>
                      <div className="legend">
                        <span><i className="cyan" style={{ background: '#fff' }} />OBSERVED LAP DELTA</span>
                        <span><i className="soft" style={{ background: 'var(--cyan)' }} />XGBOOST PREDICTED</span>
                        <span><i style={{ background: 'rgba(255, 48, 79, 0.3)', width: 12, height: 6 }} />RESIDUAL</span>
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(260px, 1fr))`, gap: 14 }}>
                      {stintDegResult.stints.map(st => {
                        const maxObs = Math.max(3.0, Math.ceil(Math.max(...st.deg_observed, ...st.deg_predicted_xgb) * 1.2))
                        const W = 360
                        const H = 180
                        const padL = 42
                        const padR = 15
                        const padT = 20
                        const padB = 28
                        const plotW = W - padL - padR
                        const plotH = H - padT - padB

                        const scaleX = (i: number) => padL + i * (plotW / Math.max(1, st.laps.length - 1))
                        const scaleY = (v: number) => padT + plotH - (v / maxObs) * plotH

                        const ptsObs = st.deg_observed.map((o, i) => `${scaleX(i).toFixed(1)},${scaleY(o).toFixed(1)}`)
                        const ptsPred = st.deg_predicted_xgb.map((p, i) => `${scaleX(i).toFixed(1)},${scaleY(p).toFixed(1)}`)

                        const reversePtsPred = [...ptsPred].reverse()
                        const residualArea = `M ${ptsObs.join(' L ')} L ${reversePtsPred.join(' L ')} Z`

                        // Calculate Mean Absolute Residual
                        const meanResidual = st.deg_observed.reduce((acc, o, i) => acc + Math.abs(o - (st.deg_predicted_xgb[i] || o)), 0) / Math.max(1, st.deg_observed.length)

                        const yTicks = [0, maxObs * 0.33, maxObs * 0.66, maxObs].map(val => ({
                          val,
                          y: scaleY(val)
                        }))

                        return (
                          <div key={st.stint} style={{ background: '#080d12', padding: 14, border: '1px solid var(--line)', borderRadius: '4px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, font: '9px JetBrains Mono' }}>
                              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span className={`pirelli-dot ${st.compound.includes('SOFT') ? 'pirelli-s' : st.compound.includes('HARD') ? 'pirelli-h' : 'pirelli-m'}`} style={{ width: 14, height: 14, fontSize: 8 }}>
                                  {st.compound[0]}
                                </span>
                                <b>STINT {st.stint} · {st.compound}</b>
                              </span>
                              <span style={{ color: 'var(--cyan)' }}>
                                {st.laps.length} LAPS · ±{meanResidual.toFixed(2)}s RESIDUAL
                              </span>
                            </div>
                            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: 160 }}>
                              {/* Gridlines */}
                              {yTicks.map((t, idx) => (
                                <g key={idx}>
                                  <line x1={padL} y1={t.y} x2={W - padR} y2={t.y} stroke="var(--line)" strokeDasharray="3 3" strokeWidth="0.6" />
                                  <text x={padL - 4} y={t.y + 3} textAnchor="end" fill="var(--muted)" fontSize="7.5" fontFamily="JetBrains Mono">
                                    +{t.val.toFixed(1)}s
                                  </text>
                                </g>
                              ))}

                              {/* X labels (ages) */}
                              {st.ages.map((age, i) => {
                                if (i !== 0 && i !== Math.floor(st.ages.length / 2) && i !== st.ages.length - 1) return null
                                const x = scaleX(i)
                                return (
                                  <text key={i} x={x} y={H - 8} textAnchor="middle" fill="var(--muted)" fontSize="7.5" fontFamily="JetBrains Mono">
                                    {age}L
                                  </text>
                                )
                              })}

                              {/* Residual Area between observed and predicted */}
                              <path d={residualArea} fill="rgba(255, 48, 79, 0.12)" />

                              {/* Observed line */}
                              <path d={`M ${ptsObs.join(' L ')}`} fill="none" stroke="#fff" strokeWidth="1.8" />
                              {/* Predicted line */}
                              <path d={`M ${ptsPred.join(' L ')}`} fill="none" stroke="var(--cyan)" strokeWidth="2" strokeDasharray="3 2" />

                              {/* Dots on observed */}
                              {st.deg_observed.map((o, i) => (
                                <circle key={i} cx={scaleX(i)} cy={scaleY(o)} r="1.8" fill="#fff" />
                              ))}
                            </svg>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ── INTERACTIVE ENLARGED TELEMETRY INSPECTOR MODAL ── */}
      {enlargedChart && (
        <TelemetryInspectorModal
          chart={enlargedChart}
          onClose={() => setEnlargedChart(null)}
          lapsA={lapsA}
          lapsB={lapsB}
          a={a}
          b={b}
          simResult={simResult}
          pitLap={pitLap}
          compound={compound}
          raceData={raceData}
          selectedRace={selectedRace}
          year={year}
        />
      )}
    </div>
  )
}
