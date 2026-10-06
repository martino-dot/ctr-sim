#!/usr/bin/env python3
"""
Re-verify ctr_teleop.py against the ctr-sim repo (run from the folder containing ctr_teleop.py, with ctr_sim installed).

  1. API / convention checks against ctr_sim (resolved_rate_step_v2 signature, insertion intervals, J = 2I, robot validation)
  2. An independent reference torsion solver (same ODE as ctr_sim.torsion_v2, but shooting BACKWARD from the free tips, which
     is well conditioned) is cross-checked against ctr_sim's own base torsional strains
  3. SnapGuard limits vs the true fold angle of the full 3-tube robot for several rotation patterns
     (PASS = guard stops before the fold; ratio = guard limit / true fold)

Usage:  python verify_snapguard.py          (~2 min, 3 processes)
        python verify_snapguard.py --full   (more patterns, several minutes)
"""
import inspect, os, sys, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import ctr_teleop as m
from ctr_sim import Material, Tube, CTRState, ConcentricTubeRobot
from ctr_sim.control.resolved_rate import resolved_rate_step_v2
from ctr_sim.kinematics.intervals import occupied_intervals
import ctr_sim.mechanics.torsion_v2 as tv

# ---------------------------------------------------------------- reference solver

class Ref:
    def __init__(self, specs, E, G, z, ds=0.002):
        self.kap=np.array([s["kappa"] for s in specs]); I=np.pi/64*np.array([s["od"]**4-s["id"]**4 for s in specs])
        self.EI=E*I; self.GJ=G*2*I; self.L=np.array([s["length"] for s in specs]); self.z=np.array(z,float); self.ds=ds
        self.n=len(specs)
    def _rhs(self, th, dth, act):
        acc=np.zeros(len(act))
        for a,i in enumerate(act):
            m=0.0
            for b,j in enumerate(act):
                if j!=i: m+=self.EI[j]*self.kap[i]*self.kap[j]*np.sin(th[a]-th[b])
            acc[a]=m/self.GJ[i]
        return dth, acc
    def _step_back(self, th, dth, act, s_from, s_to):
        L=s_from-s_to
        if L<=1e-12 or not act: return th, dth
        n=max(int(np.ceil(L/self.ds)),2); h=-L/n
        for _ in range(n):
            k1=self._rhs(th,dth,act)
            k2=self._rhs(th+0.5*h*k1[0],dth+0.5*h*k1[1],act)
            k3=self._rhs(th+0.5*h*k2[0],dth+0.5*h*k2[1],act)
            k4=self._rhs(th+h*k3[0],dth+h*k3[1],act)
            th=th+h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); dth=dth+h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        return th, dth
    def base(self, psi):
        """psi: tip angle of every tube (n,). Returns theta(0), theta'(0) for every tube."""
        order=np.argsort(-self.z); act=[]; th=np.zeros(0); dth=np.zeros(0); s=self.z[order[0]]
        for k in order:
            th,dth=self._step_back(th,dth,act,s,self.z[k]); s=self.z[k]
            act.append(k); th=np.append(th,psi[k]); dth=np.append(dth,0.0)
        th,dth=self._step_back(th,dth,act,s,0.0)
        T0=np.zeros(self.n); D0=np.zeros(self.n)
        for a,i in enumerate(act): T0[i]=th[a]; D0[i]=dth[a]
        return T0, D0
    def motor(self, psi, transmission):
        T0,D0=self.base(psi)
        return T0 - (D0*(self.L-self.z) if transmission else 0.0), D0

def rel(a):  # angles relative to the last tube
    return a[:-1]-a[-1]

def follow(ref, direction, transmission, max_deg=720, step=1.0):
    """Rotate tube `rotor` from the unwound state; return (fold_deg or None). psi unknowns: tip angles relative to last tube."""
    n=ref.n; psi=np.zeros(n)
    f=lambda p: rel(ref.motor(np.append(p,0.0),transmission)[0])
    p=np.zeros(n-1); deg=0.0; detJ_prev=None
    def jac(p):
        f0=f(p); J=np.zeros((n-1,n-1)); e=1e-6
        for c in range(n-1):
            q=p.copy(); q[c]+=e; J[:,c]=(f(q)-f0)/e
        return J,f0
    while deg<max_deg:
        d=np.radians(deg+step)*np.asarray(direction,float); tgt=rel(d)
        q=p.copy(); ok=False
        for it in range(25):
            J,f0=jac(q); r=f0-tgt
            if np.linalg.norm(r)<1e-9: ok=True; break
            try: dq=np.linalg.solve(J,-r)
            except np.linalg.LinAlgError: break
            q=q+np.clip(dq,-0.5,0.5)
        J,_=jac(q) if ok else (None,None)
        if not ok: return deg
        det=np.linalg.det(J)
        if det<=1e-6*max(1.0,abs(detJ_prev or 1.0)) : return deg+step    # slope of alpha(psi) vanished -> fold
        detJ_prev=det; p=q; deg+=step
    return None


# ---------------------------------------------------------------- checks
def build_robot(z, rot):
    nit = Material("Nitinol", m.YOUNGS_MODULUS, m.SHEAR_MODULUS)
    tubes = [Tube(s["name"], s["length"], s["kappa"], s["od"], s["id"], nit) for s in m.TUBE_SPECS]
    return ConcentricTubeRobot(tubes, CTRState(list(z), list(rot)))


def check_api():
    ok = True
    params = list(inspect.signature(resolved_rate_step_v2).parameters)
    c = "dx" in params and "jacobian" in params
    print(f"  resolved_rate_step_v2 params {params}: {'PASS' if c else 'FAIL'}"); ok &= c
    z = m.INITIAL_Q[:3]; r = build_robot(z, [0, 0, 0])
    iv = occupied_intervals(r)
    c = all(abs(a - (zi - t.length)) < 1e-12 and abs(b - zi) < 1e-12 for (a, b), zi, t in zip(iv, z, r.tubes))
    print(f"  insertion convention, tube i occupies [beta-L, beta]: {'PASS' if c else 'FAIL'}"); ok &= c
    c = all(abs(t.J - 2 * t.I) < 1e-18 for t in r.tubes)
    print(f"  J == 2I (guard uses GJ = G*2I): {'PASS' if c else 'FAIL'}"); ok &= c
    c = np.all(m.Q_MAX[:3] == [t.length for t in r.tubes]) and np.all(m.INITIAL_Q[:3] <= m.Q_MAX[:3])
    print(f"  joint limits 0 <= beta <= L match robot.validate_configuration: {'PASS' if c else 'FAIL'}"); ok &= c
    return ok


def check_reference():
    from scipy.optimize import fsolve
    z = tuple(m.INITIAL_Q[:3]); rot = [0, 0, 0.01]
    res = tv.solve_torsion_shooting(build_robot(z, rot))
    refs = Ref(m.TUBE_SPECS, m.YOUNGS_MODULUS, m.SHEAR_MODULUS, z)
    f = lambda p: rel(refs.motor(np.append(p, 0.0), False)[0]) - rel(np.array(rot))
    p = fsolve(f, np.zeros(2), xtol=1e-12); _, D0 = refs.base(np.append(p, 0.0))
    err = np.abs(D0 - res.x).max()
    print(f"  base torsional strain: ctr_sim {np.round(res.x, 4)} vs reference {np.round(D0, 4)} (max err {err:.1e}): "
          f"{'PASS' if res.success and err < 1e-3 else 'FAIL'}")
    return res.success and err < 1e-3


def true_fold(direction):
    refs = Ref(m.TUBE_SPECS, m.YOUNGS_MODULUS, m.SHEAR_MODULUS, tuple(m.INITIAL_Q[:3]), ds=0.0025)
    return follow(refs, np.asarray(direction, float), False, max_deg=420, step=2.0)


PATTERNS = [("inner only", (0, 0, 1)), ("middle only", (0, 1, 0)), ("alternating +-+", (1, -1, 1))]
FULL_EXTRA = [("outer only", (1, 0, 0)), ("outer vs inner", (1, 0, -1)), ("middle vs inner", (0, 1, -1)), ("+ + -", (1, 1, -1))]


def main():
    patterns = PATTERNS + (FULL_EXTRA if "--full" in sys.argv else [])
    print("1) ctr_sim API / conventions"); ok = check_api()
    print("2) reference solver vs ctr_sim"); ok &= check_reference()
    print("3) guard vs true 3-tube fold angle (insertions %s mm)" % np.round(m.INITIAL_Q[:3] * 1000))
    g = m.SnapGuard(m.TUBE_SPECS)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(4, len(patterns))) as ex:
        folds = list(ex.map(true_fold, [d for _, d in patterns]))
    for (name, d), fold in zip(patterns, folds):
        q0 = m.INITIAL_Q.copy(); lam = 0.0
        while lam < 720:
            q = q0.copy(); q[3:] = np.radians(lam + 1) * np.asarray(d, float)
            if g.margin(q)[0] < 0:
                break
            lam += 1
        if fold is None:
            print(f"  {name:16s} no fold up to 420 deg; guard stops at {lam:.0f} deg: PASS")
        else:
            c = lam < fold; ok &= c
            print(f"  {name:16s} true fold {fold:5.0f} deg, guard stops at {lam:4.0f} deg, ratio {lam / fold:.2f}: {'PASS' if c else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}   ({time.time() - t0:.0f} s for step 3)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
