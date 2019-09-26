import numpy as np


def princip(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi


def Rzyx(phi, theta, psi):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth]),
        np.hstack([spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi]),
        np.hstack([-sth, cth*sphi, cth*cphi])
    ])


def Tzyx(phi, theta, psi):
    sphi = np.sin(phi)
    tth = np.tan(theta)
    cphi = np.cos(phi)
    cth = np.cos(theta)

    return np.vstack([[1, sphi*tth, cphi*tth], 
                      [0, cphi, -sphi],
                      [0, sphi/cth, cphi/cth]])


def J(eta):
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    R = Rzyx(phi, theta, psi)
    T = Tzyx(phi, theta, psi)
    zero = np.zeros((3,3))

    J = np.vstack([np.hstack([R, zero]),
                   np.hstack([zero, T])])
    return J
