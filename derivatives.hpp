#pragma once

#include "accessor.hpp"

// ==============================
// First derivative coefficients
// ==============================
constexpr float L1 =  0.8f;
constexpr float L2 = -0.2f;
constexpr float L3 =  0.0380952380952381f;
constexpr float L4 = -0.0035714285714285713f;

// =========================================
// Cross second derivative coefficients
// =========================================
constexpr float L11 =  0.64f;
constexpr float L12 = -0.16f;
constexpr float L13 =  0.03047619047619047618f;
constexpr float L14 = -0.00285714285714285713f;
constexpr float L22 =  0.04f;
constexpr float L23 = -0.00761904761904761904f;
constexpr float L24 =  0.00071428571428571428f;
constexpr float L33 =  0.00145124716553287981f;
constexpr float L34 = -0.00013605442176870748f;
constexpr float L44 =  0.00001275510204081632f;

// ==============================
// Second derivative coefficients
// ==============================
constexpr float K0 = -2.84722222222222222222f;
constexpr float K1 =  1.6f;
constexpr float K2 = -0.2f;
constexpr float K3 =  0.02539682539682539682f;
constexpr float K4 = -0.00178571428571428571f;

// ======================================================
// First derivative (8th order)
// ======================================================
template <typename VIEW>
INLINE
float Der1(const VIEW& p, int i, int s, float dinv)
{
    using A = accessor<VIEW>;

    return (
        L1 * (A::get(p, i + s)     - A::get(p, i - s)) +
        L2 * (A::get(p, i + 2*s)   - A::get(p, i - 2*s)) +
        L3 * (A::get(p, i + 3*s)   - A::get(p, i - 3*s)) +
        L4 * (A::get(p, i + 4*s)   - A::get(p, i - 4*s))
    ) * dinv;
}

// ======================================================
// Second derivative (8th order)
// ======================================================
template <typename VIEW>
INLINE
float Der2(const VIEW& p, int i, int s, float d2inv)
{
    using A = accessor<VIEW>;

    return (
        K0 * A::get(p, i) +
        K1 * (A::get(p, i + s)     + A::get(p, i - s)) +
        K2 * (A::get(p, i + 2*s)   + A::get(p, i - 2*s)) +
        K3 * (A::get(p, i + 3*s)   + A::get(p, i - 3*s)) +
        K4 * (A::get(p, i + 4*s)   + A::get(p, i - 4*s))
    ) * d2inv;
}

// ======================================================
// Cross derivative (8th order)
// ======================================================
template <typename VIEW>
INLINE
float DerCross(const VIEW& p, int i,
               int s11, int s21, float dinv)
{
    using A = accessor<VIEW>;

    return (
        L11 * ( A::get(p,i+s21+s11) - A::get(p,i+s21-s11)
              - A::get(p,i-s21+s11) + A::get(p,i-s21-s11) )

      + L12 * ( A::get(p,i+s21+2*s11) - A::get(p,i+s21-2*s11)
              - A::get(p,i-s21+2*s11) + A::get(p,i-s21-2*s11)
              + A::get(p,i+2*s21+s11) - A::get(p,i+2*s21-s11)
              - A::get(p,i-2*s21+s11) + A::get(p,i-2*s21-s11) )

      + L13 * ( A::get(p,i+s21+3*s11) - A::get(p,i+s21-3*s11)
              - A::get(p,i-s21+3*s11) + A::get(p,i-s21-3*s11)
              + A::get(p,i+3*s21+s11) - A::get(p,i+3*s21-s11)
              - A::get(p,i-3*s21+s11) + A::get(p,i-3*s21-s11) )

      + L14 * ( A::get(p,i+s21+4*s11) - A::get(p,i+s21-4*s11)
              - A::get(p,i-s21+4*s11) + A::get(p,i-s21-4*s11)
              + A::get(p,i+4*s21+s11) - A::get(p,i+4*s21-s11)
              - A::get(p,i-4*s21+s11) + A::get(p,i-4*s21-s11) )

      + L22 * ( A::get(p,i+2*s21+2*s11) - A::get(p,i+2*s21-2*s11)
              - A::get(p,i-2*s21+2*s11) + A::get(p,i-2*s21-2*s11) )

      + L23 * ( A::get(p,i+2*s21+3*s11) - A::get(p,i+2*s21-3*s11)
              - A::get(p,i-2*s21+3*s11) + A::get(p,i-2*s21-3*s11)
              + A::get(p,i+3*s21+2*s11) - A::get(p,i+3*s21-2*s11)
              - A::get(p,i-3*s21+2*s11) + A::get(p,i-3*s21-2*s11) )

      + L24 * ( A::get(p,i+2*s21+4*s11) - A::get(p,i+2*s21-4*s11)
              - A::get(p,i-2*s21+4*s11) + A::get(p,i-2*s21-4*s11)
              + A::get(p,i+4*s21+2*s11) - A::get(p,i+4*s21-2*s11)
              - A::get(p,i-4*s21+2*s11) + A::get(p,i-4*s21-2*s11) )

      + L33 * ( A::get(p,i+3*s21+3*s11) - A::get(p,i+3*s21-3*s11)
              - A::get(p,i-3*s21+3*s11) + A::get(p,i-3*s21-3*s11) )

      + L34 * ( A::get(p,i+3*s21+4*s11) - A::get(p,i+3*s21-4*s11)
              - A::get(p,i-3*s21+4*s11) + A::get(p,i-3*s21-4*s11)
              + A::get(p,i+4*s21+3*s11) - A::get(p,i+4*s21-3*s11)
              - A::get(p,i-4*s21+3*s11) + A::get(p,i-4*s21-3*s11) )

      + L44 * ( A::get(p,i+4*s21+4*s11) - A::get(p,i+4*s21-4*s11)
              - A::get(p,i-4*s21+4*s11) + A::get(p,i-4*s21-4*s11) )
    ) * dinv;
}
