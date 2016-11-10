#!/usr/bin/python2

import numpy as np
from subprocess import call
import os
import sys
import argparse
from shutil import rmtree

def Gen_H_chain_xyz(r, n, filename, bohr=False):
    with open(filename, 'w') as f:
        comment = "Hydrogen chain with {} atoms at r = {}".format(n, r)
        factor = 0.52917721092 if bohr else 1.0
        print >>f, n
        print >>f, comment
        for i in range(n):
            print >>f, "H\t0\t0\t%.10f" % (r*i*factor)

def collect_energy(filename):
    data = np.loadtxt(filename)
    return data[-1]

def run_GW(xyz_file, basis, gwsc, hartree_fock, beta, mode, nfreq, gfpower, gfuni, log=None):
    if not log:
        log = os.devnull
    logf = open(log, 'w')
    call([gwsc, "--BASIS="+basis, "--XYZ="+xyz_file, "--libint="+hartree_fock, "--MODE="+mode, "--NMATSUBARA=%d" % (nfreq), "--GFPOWER=%d" % gfpower, "--GFUNIFORM=%d" % gfuni], shell=False, stdout=logf, stderr=logf)
    if log:
        logf.close()
    return collect_energy("Energy_"+mode+".dat")

def run_H_point(n, r, basis="sto6g", mode="GW", basename=None, basedir="/tmp", beta=100, nfreq=1024, gfpower=12, gfuni=256, save=False, bohr=True, gwsc="gwsc", hartree_fock="hartree-fock"):
    if not basename:
        basename = "H%d_r%.2f_n%d_beta%d_%s_%s" % (n, r, nfreq, beta,  basis, mode)
    cur_dir = os.getcwd()
    logfile = cur_dir+'/'+basename+'.log'
    work_dir = '/'.join([basedir, basename])
    xyz_file = basename + ".xyz"
    os.system("mkdir -p " + work_dir)
    os.chdir(work_dir)
    Gen_H_chain_xyz(r, n, xyz_file, bohr)
    energy = run_GW(xyz_file, basis, gwsc=gwsc, hartree_fock=hartree_fock, beta=beta, nfreq=nfreq, mode=mode, gfpower=gfpower, gfuni=gfuni, log=logfile)
    os.chdir(cur_dir)
    if not save:
        rmtree(work_dir)
    return energy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run selfconsistent GW on Hydrogen chain" )
    parser.add_argument("-n", help="Number of Hydrogen atoms", type=int, default=2)
    parser.add_argument("--rmin", help="Minimum distance between atoms (in Angstrom)", type=float, default=1.0)
    parser.add_argument("--rmax", help="Maximum distance between atoms (in Angstrom)", type=float, default=8.0)
    parser.add_argument("--rstep", help="Step between distances", type=float, default=0.1)
    parser.add_argument("-b", "--basis", help="Basis set description", default="sto-3g")
    parser.add_argument("-e", "--executable", help="Alternative GWSC executable location", default="gwsc")
    parser.add_argument("-i", "--libint", help="Alternative libint Hartree-Fock utility location", default="hartree-fock")
    parser.add_argument("-m", "--mode", help="running mode", default="GW")
    parser.add_argument("--beta", help="Beta", type=float, default=100)
    parser.add_argument("--save", help="Save working directory after finish", action='store_true')
    parser.add_argument("--bohr", help="Turn on length unit as Bohr", action="store_true")
    parser.add_argument("--nfreq", help="number of matsubara freqs", type=int, default=1024)
    parser.add_argument("--gfpower", help="number of gf power mesh", type=int, default=12)
    parser.add_argument("--gfuni", help="number of gf uniform mesh", type=int, default=256)
    parser.add_argument("-o", "--out", help="Output filename")
    parser.add_argument("--prec", help="Dyson precision", type=float, default=1e-7)

    args = parser.parse_args()
    cur_dir = os.getcwd()
    fout = sys.stdout
    if args.out:
        fout = open(args.out, 'a')
    r = args.rmin
    if args.rstep < 0:
        r = args.rmax
    print args.rstep
    while r <= args.rmax and r >= args.rmin:
        save = True if args.save else False
        bohr = True if args.bohr else False
        energy = run_H_point(args.n, r, basis=args.basis, mode=args.mode, gwsc=args.executable, hartree_fock=args.libint, basedir=cur_dir, beta=args.beta, save=save, bohr=bohr, nfreq=args.nfreq, gfpower=args.gfpower, gfuni=args.gfuni)
        print r, energy
        if args.out:
            print >>fout, r, energy
        r += args.rstep
    if args.out:
        fout.close()

