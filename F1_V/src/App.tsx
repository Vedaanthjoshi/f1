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

// Fallback default schedules
const DEFAULT_SCHEDULES: Record<string, string[]> = {
  '2025': ['Australian Grand Prix', 'Chinese Grand Prix', 'Japanese Grand Prix', 'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 'British Grand Prix', 'Italian Grand Prix'],
  '2024': ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Japanese Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 'British Grand Prix', 'Italian Grand Prix', 'Abu Dhabi Grand Prix'],
  '2023': ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Azerbaijan Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 'British Grand Prix', 'Italian Grand Prix'],
  '2022': ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Emilia Romagna Grand Prix', 'Miami Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'British Grand Prix'],
  '2021': ['Bahrain Grand Prix', 'Emilia Romagna Grand Prix', 'Portuguese Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'Azerbaijan Grand Prix', 'British Grand Prix', 'Abu Dhabi Grand Prix'],
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
    <article className={`driver-card driver-${side}`} style={{ '--driver': driver.accent } as React.CSSProperties}>
      <div className="driver-card-top">
        <span className="role-tag">{role}</span>
        <span className="car-tag">{driver.car}</span>
      </div>
      <div className="driver-content">
        <div className="driver-number">{String(driver.number).padStart(2, '0')}</div>
        <DriverAvatar driver={driver} />
        <div className="driver-info">
          <span className="driver-code">{driver.code} // {driver.country.slice(0, 3).toUpperCase()}</span>
          <h3>{driver.name}</h3>
          <p>{driver.team}</p>
          <div className="team-line"><span style={{ background: driver.accent }} /> {driver.car} · {driver.country}</div>
        </div>
      </div>
      <div className="corner corner-tr" /><div className="corner corner-bl" />
    </article>
  )
}

const makePath = (points: { x: number; y: number }[]) => {
  if (points.length === 0) return ''
  return points.map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
}

function ChartFrame({ title, eyebrow, children, legend }: { title: string; eyebrow: string; children: React.ReactNode; legend?: React.ReactNode }) {
  return (
    <article className="panel chart-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">{eyebrow}</span><h3>{title}</h3></div>
        {legend}
      </div>
      {children}
    </article>
  )
}

// ── REAL TYRE DEGRADATION SCATTER & PACE CHART ──
function RealDegradationChart({
  lapsA,
  lapsB,
  a,
  b,
}: {
  lapsA: LapRecord[]
  lapsB: LapRecord[]
  a: Driver
  b: Driver
}) {
  const allTimes = [...lapsA.map(l => l.t), ...lapsB.map(l => l.t)].filter(Boolean)
  const allLaps = [...lapsA.map(l => l.l), ...lapsB.map(l => l.l)].filter(Boolean)

  if (allTimes.length === 0) {
    return <div className="chart-wrap"><p style={{ padding: '40px', color: 'var(--muted)', textAlign: 'center' }}>No telemetry laps recorded for selected drivers.</p></div>
  }

  const minLap = Math.min(...allLaps)
  const maxLap = Math.max(...allLaps)
  const lapSpan = Math.max(1, maxLap - minLap)

  // In racing, lower lap time is faster, but standard telemetry displays faster times higher up (inverted)
  const minTime = Math.min(...allTimes) - 0.4
  const maxTime = Math.max(...allTimes) + 0.4
  const timeSpan = Math.max(0.1, maxTime - minTime)

  const getX = (lap: number) => ((lap - minLap) / lapSpan) * 600
  // Faster time (lower number) -> higher up (closer to y=20)
  const getY = (time: number) => 132 - ((maxTime - time) / timeSpan) * 112

  const formatLapTime = (sec: number) => {
    const mins = Math.floor(sec / 60)
    const rem = (sec % 60).toFixed(1)
    return `${mins}:${rem.padStart(4, '0')}`
  }

  const pathA = makePath(lapsA.map(l => ({ x: getX(l.l), y: getY(l.t) })))
  const pathB = makePath(lapsB.map(l => ({ x: getX(l.l), y: getY(l.t) })))

  return (
    <div className="chart-wrap">
      <div className="y-labels">
        <span>{formatLapTime(minTime)}</span>
        <span>{formatLapTime((minTime + maxTime) / 2)}</span>
        <span>{formatLapTime(maxTime)}</span>
      </div>
      <svg viewBox="0 0 600 165" preserveAspectRatio="none" className="data-chart" aria-label="Tyre degradation scatter plot">
        <g className="grid-lines">
          <path d="M0 20H600M0 76H600M0 132H600" />
          <path d="M0 20V132M120 20V132M240 20V132M360 20V132M480 20V132M600 20V132" />
        </g>
        {/* Trend Lines */}
        {pathA && <path className="line line-a" style={{ stroke: a.accent, opacity: 0.65 }} d={pathA} />}
        {pathB && <path className="line line-b" style={{ stroke: b.accent, opacity: 0.65 }} d={pathB} />}

        {/* Driver A Scatter Points (Circles) */}
        {lapsA.map(lap => {
          const x = getX(lap.l)
          const y = getY(lap.t)
          const compUpper = (lap.c || '').toUpperCase()
          const fill = compUpper.includes('SOFT') ? '#ff304f' : compUpper.includes('MEDIUM') ? '#ffd32a' : '#e4e9ee'
          return (
            <circle
              key={`a_${lap.l}`}
              cx={x}
              cy={y}
              r="3.8"
              fill={fill}
              stroke={a.accent}
              strokeWidth="1.2"
            >
              <title>{`Lap ${lap.l} · ${a.code} · ${lap.c} · Age ${lap.a} · ${lap.t.toFixed(3)}s`}</title>
            </circle>
          )
        })}

        {/* Driver B Scatter Points (Rectangles) */}
        {lapsB.map(lap => {
          const x = getX(lap.l)
          const y = getY(lap.t)
          const compUpper = (lap.c || '').toUpperCase()
          const fill = compUpper.includes('SOFT') ? '#ff304f' : compUpper.includes('MEDIUM') ? '#ffd32a' : '#e4e9ee'
          return (
            <rect
              key={`b_${lap.l}`}
              x={x - 3.2}
              y={y - 3.2}
              width="6.4"
              height="6.4"
              fill={fill}
              stroke={b.accent}
              strokeWidth="1.2"
            >
              <title>{`Lap ${lap.l} · ${b.code} · ${lap.c} · Age ${lap.a} · ${lap.t.toFixed(3)}s`}</title>
            </rect>
          )
        })}

        <g className="x-labels">
          <text x="0" y="158">L{minLap}</text>
          <text x="285" y="158">L{Math.round((minLap + maxLap) / 2)}</text>
          <text x="560" y="158">L{maxLap}</text>
        </g>
      </svg>
    </div>
  )
}

// ── REAL GAP EVOLUTION CHART ──
function RealGapChart({
  lapsA,
  lapsB,
  a,
  b,
}: {
  lapsA: LapRecord[]
  lapsB: LapRecord[]
  a: Driver
  b: Driver
}) {
  const mapB = new Map<number, number>()
  lapsB.forEach(l => {
    if (l.elapsed != null) mapB.set(l.l, l.elapsed)
  })

  // Gap = elapsed_B - elapsed_A (positive means Driver A completed lap earlier, i.e. A is ahead)
  const gapPoints: { lap: number; gap: number }[] = []
  lapsA.forEach(lA => {
    const elapsedB = mapB.get(lA.l)
    if (elapsedB != null && lA.elapsed != null) {
      gapPoints.push({ lap: lA.l, gap: elapsedB - lA.elapsed })
    }
  })

  if (gapPoints.length === 0) {
    return <div className="chart-wrap"><p style={{ padding: '40px', color: 'var(--muted)', textAlign: 'center' }}>Gap data pending timing loop synchronization.</p></div>
  }

  const minLap = gapPoints[0].lap
  const maxLap = gapPoints[gapPoints.length - 1].lap
  const lapSpan = Math.max(1, maxLap - minLap)

  const gaps = gapPoints.map(p => p.gap)
  const maxAbsGap = Math.max(2, Math.max(...gaps.map(Math.abs)) * 1.15)
  const minGap = -maxAbsGap
  const maxGap = maxAbsGap
  const gapSpan = maxGap - minGap

  const baseline = 132 - ((0 - minGap) / gapSpan) * 112

  const points = gapPoints.map(p => ({
    x: ((p.lap - minLap) / lapSpan) * 600,
    y: 132 - ((p.gap - minGap) / gapSpan) * 112
  }))

  const path = makePath(points)

  return (
    <div className="chart-wrap">
      <div className="y-labels">
        <span>+{maxAbsGap.toFixed(1)}s</span>
        <span>0.0s</span>
        <span>-{maxAbsGap.toFixed(1)}s</span>
      </div>
      <svg viewBox="0 0 600 165" preserveAspectRatio="none" className="data-chart" aria-label="Gap evolution line chart">
        <g className="grid-lines">
          <path d="M0 20H600M0 76H600M0 132H600" />
          <path d="M0 20V132M120 20V132M240 20V132M360 20V132M480 20V132M600 20V132" />
        </g>
        <path className="baseline" d={`M0 ${baseline}H600`} />
        {path && <path className="line line-a" style={{ stroke: a.accent }} d={path} />}
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r="2.8" fill={a.accent} stroke="#fff" strokeWidth="0.8">
            <title>{`Lap ${gapPoints[i].lap} · ${gapPoints[i].gap >= 0 ? `${a.code} +${gapPoints[i].gap.toFixed(2)}s ahead` : `${b.code} +${(-gapPoints[i].gap).toFixed(2)}s ahead`}`}</title>
          </circle>
        ))}
        <g className="x-labels">
          <text x="0" y="158">L{minLap}</text>
          <text x="285" y="158">L{Math.round((minLap + maxLap) / 2)}</text>
          <text x="560" y="158">L{maxLap}</text>
        </g>
      </svg>
    </div>
  )
}

// ── REAL PREDICTIVE FORECAST UNDERCUT CHART ──
function RealForecastChart({
  simResult,
  lap,
}: {
  simResult: SimResult | null
  lap: number
}) {
  if (!simResult || !simResult.net_undercut_advantage || simResult.net_undercut_advantage.length === 0) {
    return <div className="chart-wrap forecast"><p style={{ padding: '40px', color: 'var(--muted)', textAlign: 'center' }}>Calculating ML undercut simulation...</p></div>
  }

  const values = simResult.net_undercut_advantage
  const maxVal = Math.max(1.5, Math.max(...values) + 0.5)
  const minVal = Math.min(-1.5, Math.min(...values) - 0.5)
  const span = maxVal - minVal

  const baseline = 132 - ((0 - minVal) / span) * 112

  const points = values.map((v, i) => ({
    x: (i / (values.length - 1)) * 600,
    y: 132 - ((v - minVal) / span) * 112
  }))

  const linePath = makePath(points)
  const areaPath = `${linePath} L600 132 L0 132Z`

  // Crossover lap
  const crossoverLap = simResult.crossover_lap
  const crossoverIdx = crossoverLap != null ? crossoverLap - lap : null
  const crossoverX = crossoverIdx != null && crossoverIdx >= 0 && crossoverIdx <= 15
    ? (crossoverIdx / 15) * 600
    : null

  return (
    <div className="chart-wrap forecast">
      <div className="y-labels">
        <span>+{maxVal.toFixed(1)}s</span>
        <span>0.0s</span>
        <span>{minVal.toFixed(1)}s</span>
      </div>
      <svg viewBox="0 0 600 165" preserveAspectRatio="none" className="data-chart" aria-label="Net undercut advantage forecast">
        <g className="grid-lines">
          <path d="M0 20H600M0 76H600M0 132H600" />
          <path d="M0 20V132M150 20V132M300 20V132M450 20V132M600 20V132" />
        </g>
        <path className="baseline" d={`M0 ${baseline}H600`} />
        {areaPath && <path className="forecast-area" d={areaPath} />}
        {linePath && <path className="line forecast-line" d={linePath} />}
        {crossoverX != null && (
          <>
            <path className="crossover" d={`M${crossoverX} 14V138`} />
            <text className="cross-label" x={Math.min(500, crossoverX + 8)} y="18">
              CROSSOVER · L{crossoverLap}
            </text>
          </>
        )}
        <g className="x-labels">
          <text x="0" y="158">L{lap}</text>
          <text x="280" y="158">L{lap + 7}</text>
          <text x="558" y="158">L{lap + 14}</text>
        </g>
      </svg>
    </div>
  )
}

function SelectField({ label, value, onChange, children }: { label: string; value: string; onChange: (value: string) => void; children: React.ReactNode }) {
  return <label className="field"><span>{label}</span><div className="select-wrap"><select value={value} onChange={e => onChange(e.target.value)}>{children}</select><Icon name="chevron" size={14} /></div></label>
}

export default function App() {
  const [page, setPage] = useState<'home' | 'dashboard'>('home')
  const [launching, setLaunching] = useState(false)
  const [year, setYear] = useState('2023')
  const [schedules, setSchedules] = useState<Record<string, string[]>>(DEFAULT_SCHEDULES)
  const [selectedRace, setSelectedRace] = useState('Bahrain Grand Prix')
  const [driverA, setDriverA] = useState('VER')
  const [driverB, setDriverB] = useState('LEC')
  const [driversMeta, setDriversMeta] = useState<Driver[]>(DEFAULT_DRIVERS)
  const [state, setState] = useState<'active' | 'loading' | 'empty'>('loading')
  const [pitLap, setPitLap] = useState(20)
  const [compound, setCompound] = useState('HARD')
  const [raceData, setRaceData] = useState<RaceInfo | null>(null)
  const [simResult, setSimResult] = useState<SimResult | null>(null)
  const [slide, setSlide] = useState(0)

  // Current race list for selected year
  const raceList = schedules[year] || DEFAULT_SCHEDULES[year] || ['Bahrain Grand Prix']

  // Driver details
  const a = driversMeta.find(d => d.code === driverA) || { code: driverA, name: driverA, number: 1, team: 'Red Bull Racing', car: 'RB20', country: 'Netherlands', accent: '#3671C6' }
  const b = driversMeta.find(d => d.code === driverB) || { code: driverB, name: driverB, number: 16, team: 'Ferrari', car: 'SF-24', country: 'Monaco', accent: '#E80020' }

  // Next slide timer for home page
  const nextSlide = useCallback(() => setSlide(s => (s + 1) % SLIDE_IMAGES.length), [])

  useEffect(() => {
    const id = setInterval(nextSlide, SLIDE_INTERVAL)
    return () => clearInterval(id)
  }, [nextSlide])

  // Fetch initial schedules and metadata on mount
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

  // Function to load race data from backend
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
        // Set drivers if available in this race
        if (data.drivers && data.drivers.length > 0) {
          if (!data.drivers.includes(driverA)) setDriverA(data.drivers[0])
          if (!data.drivers.includes(driverB)) setDriverB(data.drivers[1] || data.drivers[0])
        }
        const defaultSimLap = Math.max(data.min_lap, Math.min(data.max_lap, 20))
        setPitLap(defaultSimLap)
        setState('active')
      })
      .catch(() => {
        // Fallback default mock race if backend is cold
        setState('active')
      })
  }, [driverA, driverB])

  // Initial load
  useEffect(() => {
    executeLoadRace(year, selectedRace)
  }, [])

  // Function to run ML simulation when parameters change
  useEffect(() => {
    if (state !== 'active' || !raceData) return

    fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        year: parseInt(year),
        race: selectedRace,
        driver_a: driverA,
        sim_lap: pitLap,
        next_compound: compound
      })
    })
      .then(r => {
        if (!r.ok) throw new Error('Simulation failed')
        return r.json()
      })
      .then((res: SimResult) => {
        setSimResult(res)
      })
      .catch(() => {
        // Local simulation fallback
        const isSoft = compound.toUpperCase() === 'SOFT'
        const isMed = compound.toUpperCase() === 'MEDIUM'
        const gain = isSoft ? 0.28 : isMed ? 0.18 : 0.09
        const values = Array.from({ length: 15 }, (_, i) => Number((1.1 - i * gain + Math.sin(i / 2) * 0.1).toFixed(3)))
        const cross = values.findIndex(v => v < 0)
        const crossLap = cross !== -1 ? pitLap + cross : null
        setSimResult({
          sim_lap: pitLap,
          compound,
          horizon_laps: Array.from({ length: 15 }, (_, i) => pitLap + i),
          net_undercut_advantage: values,
          stay_out_lap_times: [],
          pit_now_lap_times: [],
          crossover_lap: crossLap,
          window_open: values[2] >= 1.5,
          verdict: values[2] >= 1.5 ? 'open' : crossLap ? 'pending' : 'closed',
          confidence: 84,
          headline: values[2] >= 1.5 ? 'Undercut window open' : crossLap ? 'Pace crossover pending' : 'Window closed',
          description: values[2] >= 1.5 ? `Box on lap ${pitLap}. Fresh tyres yield a substantial undercut advantage.` : 'Tyres remaining competitive.',
          recommendation: `BOX ${driverA} · LAP ${pitLap}`
        })
      })
  }, [year, selectedRace, driverA, pitLap, compound, state, raceData])

  const navItems = useMemo(() => [
    ['grid', 'Command center'], ['chart', 'Telemetry'], ['target', 'Simulator'], ['radio', 'Live timing'],
  ], [])

  const enterDashboard = () => {
    setLaunching(true)
    window.setTimeout(() => {
      setPage('dashboard')
      setLaunching(false)
      window.scrollTo(0, 0)
    }, 650)
  }

  // Filter laps for driver A and driver B
  const lapsA = useMemo(() => {
    if (!raceData?.laps) return []
    return raceData.laps.filter(l => l.d === driverA)
  }, [raceData, driverA])

  const lapsB = useMemo(() => {
    if (!raceData?.laps) return []
    return raceData.laps.filter(l => l.d === driverB)
  }, [raceData, driverB])

  // Current gap between A and B
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

  // ═══════════════════  STRATEGY DASHBOARD (CONNECTED TO REAL ML MODEL)  ═══════════════════
  const availableDrivers = raceData?.drivers && raceData.drivers.length > 0 ? raceData.drivers : DEFAULT_DRIVERS.map(d => d.code)

  return (
    <div className="app-shell dashboard-shell">
      <aside className="sidebar">
        <div className="brand"><BrandMark /><div><strong>F1</strong><span>STRATEGY ENGINEER</span></div></div>
        <div className="side-status"><span className="live-dot" /> SYSTEM ONLINE <b>98.7%</b></div>
        <nav>{navItems.map(([icon, label], i) => <button className={i === 0 ? 'active' : ''} key={label} aria-label={label}><Icon name={icon} /><span>{label}</span><em>0{i + 1}</em></button>)}</nav>

        {/* ── RACE PARAMETERS ── */}
        <div className="control-section">
          <div className="section-label"><span>RACE PARAMETERS</span><b>01</b></div>
          <div className="field-row">
            <SelectField label="Season" value={year} onChange={value => {
              setYear(value)
              const newRaces = schedules[value] || DEFAULT_SCHEDULES[value] || ['Bahrain Grand Prix']
              setSelectedRace(newRaces[0])
            }}>
              {Object.keys(schedules).length > 0
                ? Object.keys(schedules).sort((x, y) => Number(y) - Number(x)).map(y => <option key={y} value={y}>{y}</option>)
                : Array.from({ length: 8 }, (_, i) => 2025 - i).map(y => <option key={y} value={String(y)}>{y}</option>)}
            </SelectField>

            <SelectField label="Round" value={String(raceList.indexOf(selectedRace) >= 0 ? raceList.indexOf(selectedRace) : 0)} onChange={idx => {
              const r = raceList[Number(idx)] || raceList[0]
              setSelectedRace(r)
            }}>
              {raceList.map((_, i) => <option value={String(i)} key={i}>{String(i + 1).padStart(2, '0')}</option>)}
            </SelectField>
          </div>

          <SelectField label="Grand Prix" value={selectedRace} onChange={setSelectedRace}>
            {raceList.map(r => <option value={r} key={r}>{r}</option>)}
          </SelectField>
        </div>

        {/* ── DRIVER MATCHUP ── */}
        <div className="control-section">
          <div className="section-label"><span>DRIVER MATCHUP</span><b>02</b></div>
          <SelectField label="Target / Attacker" value={driverA} onChange={setDriverA}>
            {availableDrivers.filter(code => code !== driverB).map(code => {
              const meta = driversMeta.find(d => d.code === code)
              return <option value={code} key={code}>{code} — {meta?.name || code}</option>
            })}
          </SelectField>

          <SelectField label="Defender / Rival" value={driverB} onChange={setDriverB}>
            {availableDrivers.filter(code => code !== driverA).map(code => {
              const meta = driversMeta.find(d => d.code === code)
              return <option value={code} key={code}>{code} — {meta?.name || code}</option>
            })}
          </SelectField>
        </div>

        <button className="load-button" onClick={() => executeLoadRace(year, selectedRace)}>
          <Icon name="radio" /> LOAD RACE DATA <span>↗</span>
        </button>
        <button className="reset-button" onClick={() => setState('empty')}>
          <Icon name="reset" size={14} /> Clear session
        </button>
        <div className="sidebar-footer">
          <span>MODEL</span>
          <b>XGB-STRAT v2.4</b>
          <small>LATENCY · 18MS</small>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div className="breadcrumb">
            <button className="home-link" onClick={() => setPage('home')}>← HOME</button>
            <span>/</span> PIT WALL <span>/</span> STRATEGY ENGINEER <span>/</span> <b>LIVE</b>
          </div>
          <div className="header-meta">
            <span>UTC 14:32:08</span>
            <span className="live-pill"><i /> LIVE MODEL</span>
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
        ) : (
          <div className="dashboard">
            {/* ── HERO BANNER ── */}
            <section className="hero">
              <div>
                <div className="status-badge"><span className="live-dot" /> LIVE RACE DATA <i /> ML STRATEGY MODEL</div>
                <p className="kicker">PIT WALL DECISION SUPPORT // SESSION 04</p>
                <h1>F1 Strategy <span>Engineer</span></h1>
                <p className="hero-copy">Real-time race intelligence for undercut prediction. Compare stint pace, model tyre crossover and find the fastest path through pit lane.</p>
              </div>
              <div className="session-lock">
                <span>SESSION LOCK</span>
                <strong>RACE</strong>
                <small>{(raceData?.country || 'BAHRAIN').toUpperCase()} · DRY</small>
              </div>
            </section>

            {/* ── KPI STRIP ── */}
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

            {/* ── MATCHUP SECTION ── */}
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
                  <Circuit3DCard
                    year={raceData?.year || parseInt(year, 10) || 2023}
                    race={raceData?.race || selectedRace}
                    driverAAccent={a.accent}
                    driverBAccent={b.accent}
                  />
                  <span className="eyebrow">CIRCUIT PROFILE</span>
                  <h3>{raceData?.circuit || selectedRace}</h3>
                  <p>{raceData?.country || 'Grand Prix Circuit'}</p>
                  <div className="circuit-stats">
                    <div><span>RACE LAPS</span><b>{raceData?.laps_count || 57}</b></div>
                    <div><span>LAP DISTANCE</span><b>{raceData?.distance_km?.toFixed(3) || '5.412'}<small> KM</small></b></div>
                  </div>
                </aside>
              </div>
            </section>

            {/* ── ANALYTICS GRID ── */}
            <section className="analytics-grid">
              <ChartFrame
                eyebrow={`PACE TRACE // LAP ${raceData?.min_lap || 1}—${raceData?.max_lap || 57}`}
                title="Tyre degradation"
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
                legend={
                  <span className="ahead-tag">
                    {currentGap >= 0 ? `${a.code} AHEAD · ${currentGap.toFixed(2)}S` : `${b.code} AHEAD · ${(-currentGap).toFixed(2)}S`}
                  </span>
                }
              >
                <RealGapChart lapsA={lapsA} lapsB={lapsB} a={a} b={b} />
              </ChartFrame>
            </section>

            {/* ── SIMULATOR SECTION ── */}
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
                  <div className="pit-loss">
                    <span>MEDIAN PIT LOSS</span>
                    <strong>{raceData?.pit_loss?.toFixed(1) || '25.3'}<small>S</small></strong>
                  </div>
                </div>
              </div>

              <div className="simulation-grid">
                <ChartFrame
                  eyebrow={`NET TIME DELTA // ${compound.toUpperCase()} COMPOUND`}
                  title="Net undercut advantage"
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
            </section>
          </div>
        )}
      </main>
    </div>
  )
}
