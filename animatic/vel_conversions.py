"""
Solar-frame lab kinematics for dark-matter direct detection.

Computes the lab's position and velocity relative to the SUN (not boosted
into the Galactic rest frame), expressed in either ecliptic-oriented or
Galactic-oriented Cartesian axes, at arbitrary times, using astropy's
ephemeris instead of the perturbative-in-eccentricity expansion of
Lee, Lisanti & Safdi (2013, arXiv:1307.5323).

Two orthogonal choices are exposed:

  REST FRAME (translation): are we at rest with the Sun, or boosted by the
  Sun's own motion through the Galaxy?
      - "solar frame"   : v_lab(t) = V_earth(t) [+ V_rotation(t)]
      - "galactic frame": v_obs(t) = v_sun + v_lab(t)     [paper's Eq. 3.1]

  AXIS ORIENTATION (rotation): which Cartesian basis are the components
  expressed in?
      - "ecliptic" : X -> mean vernal equinox (J2000), Z -> North Ecliptic
                     Pole, Y = Z x X. Static (equinox fixed at J2000, so it
                     does not precess with time). This is the natural frame
                     for Earth's orbit -- by definition Earth's ORBITAL
                     PLANE is the ecliptic, so r_sun_to_earth has ~zero
                     z-component in this frame at all times.
      - "galactic" : X -> Galactic Center, Y -> direction of disk rotation,
                     Z -> North Galactic Pole (IAU 1958 definition). This
                     is the frame the DM velocity distribution (e.g. the
                     SHM) is normally defined in.

Main entry points
-----------------
    sun_to_lab_position(time, location=None, frame='ecliptic')
        Position vector FROM the Sun TO the lab, r(t) = r_earth(t) - r_sun(t)
        [+ r_site_relative_to_earth_center(t)]. |r| ~ 1 AU.

    lab_velocity_solar_frame(time, location=None, include_rotation=False,
                              frame='ecliptic')
        v_lab(t) = V_earth(t) [+ V_rotation(t)], NO Sun-galactic-motion
        term. |v_lab| ~ 29-30 km/s (Earth's orbital speed).

    lab_velocity_galactic_frame(time, v_sun_galactic=None, location=None,
                                 include_rotation=False, frame='ecliptic')
        v_obs(t) = v_sun + v_lab(t). |v_obs| ~ 220-250 km/s. This is what
        actually enters v_min(t)/eta(v_min,t) for a DM distribution defined
        in the Galactic rest frame.

Requires: astropy (pip install astropy). For a higher-precision ephemeris,
also install jplephem and call
astropy.coordinates.solar_system_ephemeris.set('jpl') before running.
"""

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    get_body_barycentric_posvel,
    SkyCoord,
    ICRS,
    Galactic,
    HeliocentricMeanEcliptic,
    EarthLocation,
    CartesianRepresentation,
    CartesianDifferential,
)

# ---------------------------------------------------------------------------
# Default astrophysical parameters (as used in Lee, Lisanti & Safdi 2013)
# ---------------------------------------------------------------------------
# Sun's velocity in the Galactic frame = LSR rotation + Sun's peculiar motion,
# expressed in GALACTIC-ORIENTED Cartesian axes (GC, rotation-dir, NGP):
#   v_LSR      = (0, v_rot, 0),      v_rot     = 220 km/s
#   v_sun,pec  = (11, 12, 7) km/s    (Schoenrich, Binney & Dehnen 2010)
V_SUN_GALACTIC_DEFAULT = u.Quantity([11.0, 232.0, 7.0], u.km / u.s)

# Frames related to ICRS by a PURE ROTATION (same origin: the solar-system
# barycenter) -- position-difference vectors and velocity vectors both
# transform the same way (just a rotation matrix) between any pair of these.
_FRAMES = {
    "icrs": ICRS,
    "galactic": Galactic,
    "ecliptic": HeliocentricMeanEcliptic,
}


def _rotate_vector(vec_xyz, v, time, from_frame="icrs", to_frame="ecliptic"):
    """
    Rotate a Cartesian 3-vector -- a position DIFFERENCE or a velocity --
    from one frame's axis orientation to another's.

    Valid for any pair of frames sharing a common origin and related only
    by a fixed rotation: 'icrs', 'galactic', 'ecliptic' (= Barycentric Mean
    Ecliptic, equinox fixed at J2000, so it does not precess with time).
    Because there's no origin/translation difference between these frames,
    position-difference vectors and velocity vectors transform identically
    -- both are just multiplied by the same rotation matrix.

    Parameters
    ----------
    vec_xyz : Quantity, shape (3,) or (3, N)
    from_frame, to_frame : {'icrs', 'galactic', 'ecliptic'}

    Returns
    -------
    Quantity, same shape and unit as input.
    """
    if from_frame == to_frame:
        return vec_xyz, v
    diff = CartesianDifferential(v)
    rep = CartesianRepresentation(vec_xyz, differentials=diff)
    
    c_src = SkyCoord(rep, frame=_FRAMES[from_frame](
        representation_type=CartesianRepresentation,
        differential_type=CartesianDifferential))
    c_dst = c_src.transform_to(_FRAMES[to_frame](obstime=time))

    return c_dst.cartesian.xyz.to(vec_xyz.unit), c_dst.velocity.d_xyz.to(v.unit)


# ---------------------------------------------------------------------------
# Raw ephemeris state vectors, in ICRS-oriented axes
# ---------------------------------------------------------------------------
def earth_state_icrs(time):
    """
    Earth's position and velocity relative to the Sun, r_earth-r_sun and
    V_earth(t), in ICRS-oriented Cartesian axes (km, km/s).
    """
    pos_e, vel_e = get_body_barycentric_posvel("earth", time)
    #pos_s, vel_s = get_body_barycentric_posvel("sun", time)
    r = (pos_e).xyz.to(u.km)
    v = (vel_e).xyz.to(u.km / u.s)
    return r, v


def site_state_icrs(time, location):
    """
    Detector site's position and velocity relative to Earth's CENTER,
    in ICRS-oriented Cartesian axes (km, km/s). Uses GCRS, which is
    kinematically non-rotating relative to ICRS (shared axis orientation
    to sub-mm/s precision), so no further rotation is needed to treat this
    as "ICRS-oriented".
    """
    pos_gcrs, vel_gcrs = location.get_gcrs_posvel(time)
    r = pos_gcrs.xyz.to(u.km)
    v = vel_gcrs.xyz.to(u.km / u.s)
    return r, v


# ---------------------------------------------------------------------------
# Position: Sun -> lab
# ---------------------------------------------------------------------------
def sun_to_lab_position(time, location=None, frame="ecliptic"):
    """
    Position vector FROM the Sun TO the lab:

        r(t) = [r_earth(t) - r_sun(t)]  +  r_site_rel_to_earth_center(t)

    Parameters
    ----------
    time : astropy.time.Time
        Scalar or array-valued time(s).
    location : astropy.coordinates.EarthLocation, optional
        Detector site. If omitted, returns the Sun-to-Earth-CENTER vector
        (the site term is ~10^-5 of the total and often negligible).
    frame : {'ecliptic', 'galactic'}, default 'ecliptic'
        Axis orientation of the returned components.

    Returns
    -------
    r : Quantity, shape (3,) or (3, N)
        Position vector, km.
    distance : Quantity, scalar or shape (N,)
        |r(t)|, km (~1 AU, varying with Earth's orbital eccentricity).
    """
    r, v = earth_state_icrs(time)

    if location is not None:
        r_site, v_site = site_state_icrs(time, location)
        r = r + r_site
        v = v + v_site

    r_out, v_out = _rotate_vector(r, v, time, "icrs", frame)

    distance = np.linalg.norm(r_out.to(u.km).value, axis=0) * u.km
    if r_out.ndim == 1:
        distance = distance.reshape(())

    speed = np.linalg.norm(v_out.to(u.km / u.s).value, axis=0) * (u.km / u.s)
    if v_out.ndim == 1:
        speed = speed.reshape(())

    return r_out, distance, v_out, speed


# ---------------------------------------------------------------------------
# Velocity: solar frame and Galactic frame
# ---------------------------------------------------------------------------
def lab_velocity_solar_frame(time, location=None, include_rotation=False, frame="ecliptic"):
    """
    Lab velocity relative to the SUN (no Sun-galactic-motion term):

        v_lab(t) = V_earth(t)  [+ V_rotation(t)]

    Parameters
    ----------
    time : astropy.time.Time
    location : astropy.coordinates.EarthLocation, optional
        Required only if include_rotation=True.
    include_rotation : bool, default False
        If True, add the site's diurnal-rotation velocity relative to
        Earth's center (daily-modulation term). ~0.3-0.46 km/s scale.
    frame : {'ecliptic', 'galactic'}, default 'ecliptic'
        Axis orientation of the returned components.

    Returns
    -------
    v_lab : Quantity, shape (3,) or (3, N), km/s.
    speed : Quantity, scalar or shape (N,), km/s. ~29-30 km/s.
    """
    _, v = earth_state_icrs(time)

    if include_rotation:
        if location is None:
            raise ValueError("`location` is required when include_rotation=True")
        _, v_site = site_state_icrs(time, location)
        v = v + v_site

    v_out = _rotate_vector(v, time, "icrs", frame)

    speed = np.linalg.norm(v_out.to(u.km / u.s).value, axis=0) * (u.km / u.s)
    if v_out.ndim == 1:
        speed = speed.reshape(())

    return v_out, speed


def lab_velocity_galactic_frame(
    time,
    v_sun_galactic=None,
    location=None,
    include_rotation=False,
    frame="ecliptic",
):
    """
    Full Galactic-rest-frame lab velocity, v_obs(t) = v_sun + v_lab(t)
    [Eq. 3.1 of Lee, Lisanti & Safdi 2013]. This is what enters
    v_min(t)/eta(v_min,t) for a DM distribution defined in the Galactic
    rest frame (e.g. the SHM). |v_obs| ~ 220-250 km/s and does NOT depend
    on `frame` (rotating axes doesn't change a vector's length) -- only the
    reported *components* depend on `frame`.

    Parameters
    ----------
    v_sun_galactic : Quantity, shape (3,), optional
        Sun's velocity in GALACTIC-ORIENTED axes (km/s). Defaults to
        V_SUN_GALACTIC_DEFAULT = (11, 232, 7) km/s. Automatically rotated
        into `frame` before being added.
    frame : {'ecliptic', 'galactic'}, default 'ecliptic'
        Axis orientation of the returned components.
    (other parameters as in lab_velocity_solar_frame)

    Returns
    -------
    v_obs, speed : as in lab_velocity_solar_frame, boosted by v_sun.
    """
    if v_sun_galactic is None:
        v_sun_galactic = V_SUN_GALACTIC_DEFAULT
    v_sun = _rotate_vector(v_sun_galactic, time, "galactic", frame)

    v_lab, _ = lab_velocity_solar_frame(time, location, include_rotation, frame)

    v_obs = v_sun[:, None] + v_lab if v_lab.ndim > 1 else v_sun + v_lab

    speed = np.linalg.norm(v_obs.to(u.km / u.s).value, axis=0) * (u.km / u.s)
    if v_obs.ndim == 1:
        speed = speed.reshape(())

    return v_obs, speed


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: use a JPL kernel for higher precision (requires jplephem)
    # from astropy.coordinates import solar_system_ephemeris
    # solar_system_ephemeris.set('jpl')

    t0 = Time("2024-01-01T00:00:00")
    times = t0 + np.arange(0, 366) * u.day

    # --- Position: Sun -> Earth center, ecliptic-oriented axes ---
    r_ecl, dist = sun_to_lab_position(times, frame="ecliptic")
    print("Sun -> Earth position (ecliptic axes):")
    print(f"  min |r| = {dist.min():.0f}  ({dist.min().to(u.au):.5f})  "
          f"at {times[np.argmin(dist)].iso[:10]}  (perihelion, expect ~early Jan)")
    print(f"  max |r| = {dist.max():.0f}  ({dist.max().to(u.au):.5f})  "
          f"at {times[np.argmax(dist)].iso[:10]}  (aphelion, expect ~early Jul)")
    z = r_ecl[2]
    print(f"  ecliptic z-component: min={z.min():.1f}, max={z.max():.1f}"
          f"  (should be ~0 -- Earth's orbit DEFINES the ecliptic plane)")

    # --- Solar-frame velocity, ecliptic-oriented axes ---
    v_lab, speed_lab = lab_velocity_solar_frame(times, frame="ecliptic")
    print(f"\nSolar-frame |v_lab|: min={speed_lab.min():.2f}, max={speed_lab.max():.2f}"
          f"  (Earth's orbital speed, ~29-30 km/s)")

    # --- Galactic-frame velocity, expressed in ecliptic-oriented axes ---
    v_obs, speed_obs = lab_velocity_galactic_frame(times, frame="ecliptic")
    print(f"\nGalactic-frame |v_obs| (ecliptic-axis components): "
          f"min={speed_obs.min():.2f}, max={speed_obs.max():.2f}"
          f"  (expect ~219-248 km/s, same speed as galactic-oriented axes)")

    # --- Single-time example with a real detector site, daily term included ---
    site = EarthLocation(lat=51.4779 * u.deg, lon=-0.0015 * u.deg, height=45 * u.m)
    t_single = Time("2024-06-02T12:00:00")
    r_single, dist_single = sun_to_lab_position(t_single, location=site, frame="ecliptic")
    v_single, speed_single = lab_velocity_solar_frame(
        t_single, location=site, include_rotation=True, frame="ecliptic"
    )
    print(f"\nAt {t_single.iso}, ecliptic axes:")
    print(f"  r (Sun->lab)  = {r_single} , |r| = {dist_single:.0f}")
    print(f"  v_lab (solar) = {v_single} , |v_lab| = {speed_single:.3f}")