# =============================================================================
#  Dipole Background Field Operators
#
#  Helper functions for implementing dipole background magnetic field case.
#  When a dipole field is present, all radial operators need shifted powers:
#  - Poloidal (u) operators: r^n → r^(n+2)
#  - Toroidal (v) operators: r^n → r^(n+3)
#  - Magnetic field operators: varies by component
#
#  This follows Kore's implementation in operators.py where:
#  cdipole = (par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0)
#
#  References:
#  - Kore: kore-main/bin/operators.py (lines 32-34, 39-41, 60-63, etc.)
#  - Kore: kore-main/bin/submatrices.py (lines 34, 261, 295, 324, 354)
# =============================================================================

"""
    is_dipole_case(B0_type, ricb)

Check if dipole field is active, requiring special operator handling.

For dipole case to be active, ALL conditions must be met:
1. `B0_type == dipole`
2. `ricb > 0` (inner core present)

Returns `true` if dipole operators should be used, `false` otherwise.

# Examples
```julia
is_dipole_case(dipole, 0.35)  # Returns true

is_dipole_case(dipole, 0.0)  # Returns false (no inner core)

is_dipole_case(axial, 0.35)  # Returns false (not dipole)
```
"""
function is_dipole_case(B0_type::BackgroundField, ricb::Real)
    return (B0_type == dipole) && (ricb > 0)
end

"""
    radial_power_shift_poloidal(is_dipole::Bool) -> Int

Get radial power shift for poloidal (2-curl) operators.

# Returns
- `2` if dipole field present (cdipole = true)
- `0` otherwise (no shift)

# Kore Reference
Line 261 in submatrices.py:
```python
labl += ut.labelit( labl_u, section='u', rplus=2*cdipole)
```

This means:
- Axial or no field: r², r³, r⁴  (standard)
- Dipole field:       r⁴, r⁵, r⁶  (shift by +2)

# Examples
```julia
shift = radial_power_shift_poloidal(true)   # Returns 2
shift = radial_power_shift_poloidal(false)  # Returns 0

# In operator construction:
r_power = 2 + radial_power_shift_poloidal(is_dipole)  # 2 or 4
r_op = sparse_radial_operator(r_power, 0, N, ri, ro)
```
"""
function radial_power_shift_poloidal(is_dipole::Bool)
    return is_dipole ? 2 : 0
end

"""
    radial_power_shift_toroidal(is_dipole::Bool) -> Int

Get radial power shift for toroidal (1-curl) operators.

# Returns
- `3` if dipole field present (cdipole = true)
- `0` otherwise (no shift)

# Kore Reference
Line 295 in submatrices.py:
```python
labl += ut.labelit( labl_v, section='v', rplus=3*cdipole)
```

This means:
- Axial or no field: r²        (standard)
- Dipole field:       r⁵        (shift by +3)

# Physical Interpretation
The toroidal component needs a larger shift (+3 vs +2) because:
1. It's a 1-curl (not 2-curl like poloidal)
2. The dipole field scales as 1/r³
3. Combined effect requires r^(n+3) scaling

# Examples
```julia
shift = radial_power_shift_toroidal(true)   # Returns 3
shift = radial_power_shift_toroidal(false)  # Returns 0

# In operator construction:
r_power = 2 + radial_power_shift_toroidal(is_dipole)  # 2 or 5
r_op = sparse_radial_operator(r_power, 0, N, ri, ro)
```
"""
function radial_power_shift_toroidal(is_dipole::Bool)
    return is_dipole ? 3 : 0
end

"""
    radial_power_shift_magnetic_poloidal(is_dipole::Bool) -> Int

Get radial power shift for magnetic poloidal (f) field operators.

# Returns
- `2` if dipole field present (cdipole = true)
- `0` otherwise (no shift)

# Kore Reference
Line 324 in submatrices.py:
```python
labl += ut.labelit( labl_f, section='f', rplus=2*cdipole)
```

# Examples
```julia
shift = radial_power_shift_magnetic_poloidal(true)   # Returns 2
shift = radial_power_shift_magnetic_poloidal(false)  # Returns 0
```
"""
function radial_power_shift_magnetic_poloidal(is_dipole::Bool)
    return is_dipole ? 2 : 0
end

"""
    radial_power_shift_magnetic_toroidal(is_dipole::Bool) -> Int

Get radial power shift for magnetic toroidal (g) field operators.

# Returns
- `3` if dipole field present (cdipole = true)
- `0` otherwise (no shift)

# Kore Reference
Line 354 in submatrices.py:
```python
labl += ut.labelit( labl_g, section='g', rplus=3*cdipole)
```

# Examples
```julia
shift = radial_power_shift_magnetic_toroidal(true)   # Returns 3
shift = radial_power_shift_magnetic_toroidal(false)  # Returns 0
```
"""
function radial_power_shift_magnetic_toroidal(is_dipole::Bool)
    return is_dipole ? 3 : 0
end

# Operator table showing shifts for dipole field case.
#
# # Poloidal Velocity (u) Operators - shift = +2
#
# | Base (Axial) | Dipole | Shift | Usage |
# |--------------|--------|-------|-------|
# | r²D⁰         | r⁴D⁰   | +2    | Coriolis, time derivative |
# | r³D¹         | r⁵D¹   | +2    | Coriolis |
# | r⁴D²         | r⁶D²   | +2    | Coriolis |
# | r⁰D⁰         | r²D⁰   | +2    | Viscous |
# | r²D²         | r⁴D²   | +2    | Viscous |
# | r³D³         | r⁵D³   | +2    | Viscous |
# | r⁴D⁴         | r⁶D⁴   | +2    | Viscous |
# | r⁴D⁰         | r⁶D⁰   | +2    | Buoyancy |
#
# # Toroidal Velocity (v) Operators - shift = +3
#
# | Base (Axial) | Dipole | Shift | Usage |
# |--------------|--------|-------|-------|
# | r²D⁰         | r⁵D⁰   | +3    | Coriolis, time derivative |
# | r¹D⁰         | r⁴D⁰   | +3    | Coriolis coupling |
# | r²D¹         | r⁵D¹   | +3    | Coriolis coupling |
# | r⁰D⁰         | r³D⁰   | +3    | Viscous |
# | r¹D¹         | r⁴D¹   | +3    | Viscous |
# | r²D²         | r⁵D²   | +3    | Viscous |
#
# # Magnetic Poloidal (f) Operators - shift = +2
#
# | Base (Axial) | Dipole | Shift | Usage |
# |--------------|--------|-------|-------|
# | r²D⁰         | r⁴D⁰   | +2    | Time derivative |
#
# # Magnetic Toroidal (g) Operators - shift = +3
#
# | Base (Axial) | Dipole | Shift | Usage |
# |--------------|--------|-------|-------|
# | r²D⁰         | r⁵D⁰   | +3    | Time derivative |
#
# # References
#
# From Kore operators.py, the systematic pattern is:
# - Line 32-34 (u time derivative): r² → r⁴ (shift +2)
# - Line 39-41 (v time derivative): r² → r⁵ (shift +3)
# - Line 60-63 (Coriolis u): r², r³, r⁴ → r⁴, r⁵, r⁶ (shift +2)
# - Line 70-73 (Coriolis u→v): r³, r⁴ → r⁵, r⁶ (shift +2)
# - Line 96-99 (Coriolis v→u): r¹, r² → r⁴, r⁵ (shift +3)
# - Line 119-122 (Coriolis v): r² → r⁵ (shift +3)
# - Line 171-173 (Viscous u): r², r⁴, r⁵, r⁶ → r², r⁴, r⁵, r⁶ + shift +2
# - Line 189-191 (Viscous v): r⁰, r¹, r² → r³, r⁴, r⁵ (shift +3)
#
# The pattern is clear and consistent throughout.
