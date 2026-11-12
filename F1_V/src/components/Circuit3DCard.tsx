import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

type Point3 = [number, number, number]

// Clean fallback circuit loop when FastF1 geometry isn't loaded yet
const FALLBACK_POINTS: Point3[] = [
  [-0.82, 0.0, 0.35],
  [-0.55, 0.02, -0.15],
  [-0.18, 0.03, 0.08],
  [0.12, 0.05, -0.42],
  [0.52, 0.04, -0.48],
  [0.82, 0.02, -0.18],
  [0.72, 0.01, 0.24],
  [0.34, 0.0, 0.14],
  [0.14, -0.01, 0.45],
  [-0.36, 0.0, 0.48],
]

function CircuitScene({
  points,
  driverAAccent,
  driverBAccent,
}: {
  points: Point3[]
  driverAAccent: string
  driverBAccent: string
}) {
  const carARef = useRef<THREE.Mesh>(null)
  const carBRef = useRef<THREE.Mesh>(null)

  const { curve, tubeGeo, startPos } = useMemo(() => {
    const vecs = (points.length >= 6 ? points : FALLBACK_POINTS).map(
      ([x, y, z]) => new THREE.Vector3(x, y, z),
    )
    const c = new THREE.CatmullRomCurve3(vecs, true, 'catmullrom', 0.35)
    const geo = new THREE.TubeGeometry(c, 180, 0.022, 10, true)
    return {
      curve: c,
      tubeGeo: geo,
      startPos: c.getPointAt(0),
    }
  }, [points])

  useEffect(() => {
    return () => {
      tubeGeo.dispose()
    }
  }, [tubeGeo])

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const uA = (t * 0.085) % 1
    const uB = (t * 0.085 + 0.93) % 1
    if (carARef.current) {
      carARef.current.position.copy(curve.getPointAt(uA))
      carARef.current.position.y += 0.03
    }
    if (carBRef.current) {
      carBRef.current.position.copy(curve.getPointAt(uB))
      carBRef.current.position.y += 0.03
    }
  })

  return (
    <>
      <ambientLight intensity={0.75} />
      <directionalLight position={[2, 4, 2]} intensity={1.4} color="#ffffff" />
      <pointLight position={[-1.5, 1.5, -1.5]} intensity={0.8} color="#20d9ef" />

      <gridHelper args={[3.0, 14, '#1e2b38', '#101820']} position={[0, -0.09, 0]} />

      {/* Circuit ribbon */}
      <mesh geometry={tubeGeo}>
        <meshStandardMaterial
          color="#20d9ef"
          emissive="#0b5563"
          emissiveIntensity={0.65}
          roughness={0.35}
          metalness={0.4}
        />
      </mesh>

      {/* Start / Finish beacon */}
      <mesh position={[startPos.x, startPos.y + 0.035, startPos.z]}>
        <boxGeometry args={[0.045, 0.07, 0.045]} />
        <meshStandardMaterial color="#e10600" emissive="#e10600" emissiveIntensity={0.9} />
      </mesh>

      {/* Animated Driver A */}
      <mesh ref={carARef}>
        <sphereGeometry args={[0.042, 16, 16]} />
        <meshStandardMaterial
          color={driverAAccent}
          emissive={driverAAccent}
          emissiveIntensity={1.1}
        />
      </mesh>

      {/* Animated Driver B */}
      <mesh ref={carBRef}>
        <sphereGeometry args={[0.038, 16, 16]} />
        <meshStandardMaterial
          color={driverBAccent}
          emissive={driverBAccent}
          emissiveIntensity={1.0}
        />
      </mesh>

      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={1.15}
        maxDistance={3.2}
        autoRotate
        autoRotateSpeed={0.85}
        maxPolarAngle={Math.PI / 2.15}
      />
    </>
  )
}

export function Circuit3DCard({
  year,
  race,
  driverAAccent = '#E10600',
  driverBAccent = '#20D9EF',
}: {
  year: number
  race: string
  driverAAccent?: string
  driverBAccent?: string
}) {
  const [points, setPoints] = useState<Point3[]>(FALLBACK_POINTS)
  const [isRealTelemetry, setIsRealTelemetry] = useState(false)

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()
    const apiBase =
      window.location.port === '5173' ? 'http://127.0.0.1:8000' : ''

    fetch(
      `${apiBase}/api/circuit-geometry?year=${encodeURIComponent(year)}&race=${encodeURIComponent(race)}`,
      { signal: controller.signal },
    )
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled) return
        if (data && Array.isArray(data.points) && data.points.length >= 6) {
          setPoints(data.points)
          setIsRealTelemetry(data.source === 'fastf1')
        } else {
          setPoints(FALLBACK_POINTS)
          setIsRealTelemetry(false)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPoints(FALLBACK_POINTS)
          setIsRealTelemetry(false)
        }
      })

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [year, race])

  return (
    <div className="circuit-outline circuit-3d-wrap">
      <Canvas
        camera={{ position: [0, 1.45, 1.65], fov: 38 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
      >
        <CircuitScene
          points={points}
          driverAAccent={driverAAccent}
          driverBAccent={driverBAccent}
        />
      </Canvas>
      <span className="circuit-3d-hint">
        {isRealTelemetry ? '3D FASTF1 · ORBIT' : '3D MODEL · ORBIT'}
      </span>
    </div>
  )
}

