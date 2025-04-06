import numpy as np


def ssa(angle):
    """
    实现将角度规范化到-pi到pi之间
    """
    return ((angle + np.pi) % (2*np.pi)) - np.pi


def Rzyx(phi, theta, psi):
    """
    输出LAUV的六自由度旋转矩阵
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth]),
        np.hstack([spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi]),
        np.hstack([-sth, cth*sphi, cth*cphi])])


def Tzyx(phi, theta, psi):
    """
    输出LAUV的六自由度旋转矩阵的导数
    """
    sphi = np.sin(phi)
    tth = np.tan(theta)
    cphi = np.cos(phi)
    cth = np.cos(theta)

    return np.vstack([
        np.hstack([1, sphi*tth, cphi*tth]), 
        np.hstack([0, cphi, -sphi]),
        np.hstack([0, sphi/cth, cphi/cth])])


def J(eta):
    """
    输出LAUV的雅可比矩阵
    """
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    R = Rzyx(phi, theta, psi)
    T = Tzyx(phi, theta, psi)
    zero = np.zeros((3,3))

    return np.vstack([
        np.hstack([R, zero]),
        np.hstack([zero, T])])


def S_skew(a):
    """
    当输入一个三维向量a时，输出一个反对称矩阵
    一般输入的是角速度
    """
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]

    return np.vstack([
        np.hstack([0, -a3, a2]),
        np.hstack([a3, 0, -a1]),
        np.hstack([-a2, a1, 0])])


def _H(r):
    """
    输出LAUV的H矩阵
    （H矩阵是一个6x6的矩阵，用于将速度转换到全局坐标系）
    """
    I3 = np.identity(3)
    zero = np.zeros((3,3))

    return np.vstack([
        np.hstack([I3, np.transpose(S_skew(r))]),
        np.hstack([zero, I3])])


def move_to_CO(A_CG, r_g):
    """
    将一个矩阵从CG坐标系转换到CO坐标系
    """
    H = _H(r_g)
    Ht = np.transpose(H)
    A_CO = Ht.dot(A_CG).dot(H)
    return A_CO
